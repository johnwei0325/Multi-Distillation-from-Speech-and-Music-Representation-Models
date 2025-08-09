"""
    Pre-train expert for distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

from easydict import EasyDict as edict
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pretrain.multi_distiller.dataset import OnlineWaveDataset
from upstream.multi_distiller.model import MultiDistillerConfig, MultiDistillerModel
from pretrain.multi_distiller.convert_dict import convert_ssast_state_dict_to_astmodel
from transformers import AutoModel, AutoConfig, ASTConfig, AutoFeatureExtractor, WhisperModel, AutoProcessor, WavLMModel
import torchaudio 
import torchaudio.transforms as transforms
from transformers import ASTForAudioClassification
# from transformers import AutoProcessor, ASTModel
import pdb
import os
# from pretrain.multi_distiller.ssast_module import SSASTPredModule
from pretrain.multi_distiller.audio import FeatureExtractor
from pretrain.multi_distiller.ast_models import ASTModel
from pretrain.multi_distiller.disable_dropout import disable_MERT_encoder_dropout, disable_wavlm_encoder_dropout, disable_SSAST_encoder_dropout
#os.environ["TORCH_HOME"] = "/home/twsgxyc199/ycevan/ycevan/johnwei/"
# from audiossl.models.atst.atst import ATST
import fairseq
from dataclasses import dataclass

@dataclass
class UserDirModule:
    user_dir: str

class TemporalAligner(nn.Module):
    def __init__(self, max_length_in_seconds=10, input_sample_rate=16000, distilhubert_frame_shift=20, ssast_frame_shift=10):
        """
        TemporalAligner for aligning the time dimension of SSAST and distilHuBERT.
        
        Args:
            max_length_in_seconds: Maximum length for SSAST (in seconds).
            input_sample_rate: The sample rate of the input audio (default 16 kHz).
            distilhubert_frame_shift: The frame shift (in ms) for distilHuBERT features.
            ssast_frame_shift: The frame shift (in ms) for SSAST features.
        """
        super(TemporalAligner, self).__init__()

        # Compute the number of samples for SSAST's max input length
        self.max_length_in_samples = max_length_in_seconds * input_sample_rate
        
        # Frame shifts in samples for SSAST and distilHuBERT
        self.distilhubert_frame_shift_samples = int((distilhubert_frame_shift / 1000) * input_sample_rate)
        self.ssast_frame_shift_samples = int((ssast_frame_shift / 1000) * input_sample_rate)
        
        # Average pooling for temporal downsampling (matching distilHuBERT with SSAST)
        self.temporal_pooling = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, ssast_features, distilhubert_features):
        """
        Align the SSAST and distilHuBERT features.
        
        Args:
            ssast_features: The feature tensor from SSAST (batch, time, feature_dim).
            distilhubert_features: The feature tensor from distilHuBERT (batch, time, feature_dim).
            
        Returns:
            Aligned distilHuBERT features cropped and temporally downsampled.
        """
        # Step 1: Perform temporal downsampling of SSAST features
        ssast_features_pooled = self.temporal_pooling(ssast_features.transpose(1, 2)).transpose(1, 2)
        
        # Step 2: Crop distilHuBERT features if they exceed the SSAST max length
        # Determine the maximum number of frames SSAST can process (10 seconds)
        max_frames_ssast = ssast_features_pooled.shape[1]
        max_frames_distilhubert = distilhubert_features.shape[1]
        
        # Crop distilHuBERT features to match the SSAST max frames
        if max_frames_distilhubert > max_frames_ssast:
            distilhubert_features_cropped = distilhubert_features[:, :max_frames_ssast, :]
        else:
            distilhubert_features_cropped = distilhubert_features
        
        if max_frames_distilhubert < max_frames_ssast:
            ssast_features_pooled = ssast_features_pooled[:, :max_frames_distilhubert, :]
    
        
        return ssast_features_pooled, distilhubert_features_cropped

def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = True

def remap_keys(state_dict, prefix):
    """Remap keys in the state_dict to match the model's expected structure."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.backbone'):
            new_key = key.replace('module.backbone', 'encoder')
        elif key.startswith('module.head'):
            new_key = key.replace('module.head', 'projector')  # Adjust based on your model's needs
        elif key.startswith('module.'):
            new_key = key.replace('module.', '', 1)
        else:
            new_key = key
        new_state_dict[f'{prefix}.{new_key}'] = value
    return new_state_dict

def average_weights(mapped_state_dicts):
    """Averages the weights from multiple state_dicts."""
    avg_dict = collections.OrderedDict()

    keys = mapped_state_dicts[0].keys()
    for key in keys:
        weights = [sd[key] for sd in mapped_state_dicts]
        avg_dict[key] = torch.mean(torch.stack(weights), dim=0)

    return avg_dict

def rename_attention_keys_mert(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key

        # Map "attention" to "self_attn"
        new_key = new_key.replace("attention.k_proj", "self_attn.k_proj")
        new_key = new_key.replace("attention.v_proj", "self_attn.v_proj")
        new_key = new_key.replace("attention.q_proj", "self_attn.q_proj")
        new_key = new_key.replace("attention.out_proj", "self_attn.out_proj")

        # Map "layer_norm" to "self_attn_layer_norm"
        new_key = new_key.replace("layer_norm", "self_attn_layer_norm")

        # Map "feed_forward" to "fc1" and "fc2"
        new_key = new_key.replace("feed_forward.intermediate_dense", "fc1")
        new_key = new_key.replace("feed_forward.output_dense", "fc2")

        # Handle the final layer norm rename
        new_key = new_key.replace("final_self_attn_layer_norm", "final_layer_norm")

        new_state_dict[new_key] = state_dict[key]

    return new_state_dict

class UpstreamPretrainExpert(nn.Module):
    """
    The Distiller pretrain expert
    """

    def __init__(
        self, datarc, upstream_config, device="cuda", multi_gpu=False, **kwargs
    ):
        super().__init__()

        self.datarc = datarc
        self.device = device
        self.multi_gpu = multi_gpu
        self.freeze = False
        self.count_freeze = 0
        if type(upstream_config) == str:
            self.upstream_config = yaml.load(
                open(upstream_config, "r"), Loader=yaml.FullLoader
            )
            print(
                "[UpstreamPretrainExpert] - Using upstream config from:",
                upstream_config,
            )
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print(
                "[UpstreamPretrainExpert] - Using upstream config from the previous experiment."
            )
        else:
            raise ValueError

        self._get_train_dataloader()

        print("[UpstreamPretrainExpert] - Initializing model...")
        model_config = MultiDistillerConfig(self.upstream_config["multi_distiller"])
        self.model = MultiDistillerForPretrain(
            self.datarc, model_config, edict(self.upstream_config["teacher"]) ### here we get the multidistiller part and the teacher part of the file
        )

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print(
                "[UpstreamPretrainExpert] - Multi-GPU training Enabled: "
                + str(torch.cuda.device_count())
            )
        print(
            "[UpstreamPretrainExpert] - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        )

    def _get_train_dataloader(self):
        dataset = OnlineWaveDataset(
            self.upstream_config["task"],
            self.datarc["train_batch_size"],
            target_level=self.upstream_config["audio"]["target_level"],
            **self.datarc,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=1,  # for bucketing
            shuffle=True,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def load_model(self, all_states):
        if self.multi_gpu:
            self.model.module.distiller.load_state_dict(all_states["Distiller"])
        else:
            self.model.distiller.load_state_dict(all_states["Distiller"])

    # Interface
    def add_state_to_save(self, all_states):
        all_states["Distiller"] = (
            self.model.float().distiller.state_dict()
            if not self.multi_gpu
            else self.model.float().module.distiller.state_dict()
        )
        all_states["Config"] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
        """
        Args:
            data:
                [wave_input, pad_mask]

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss
        """
        # if self.freeze:
        #     self.count_freeze += 1
        #     if self.count_freeze == 500:
        #         unfreeze_model(self.model.distiller)
        #         print('Unfreezed Student')
        #         self.freeze = False
        #         self.count_freeze = 0

        # if global_step % 2500 == 0 and not self.freeze:
        #     freeze_model(self.model.distiller)
        #     unfreeze_model(self.model.distiller.translator)
        #     print('Freezed Student')
        #     self.freeze = True
        
        wave_input, wave_orig_16k, wave_orig_24k, wave_len, pad_mask = data
        wave_input = wave_input.to(self.device)
        wave_len = wave_len.to(self.device)
        pad_mask = pad_mask.type(wave_input.dtype).to(self.device)

        loss, losses, other_res = self.model(
            wave_input,
            wave_orig_16k,
            wave_orig_24k,
            wave_len,
            pad_mask,
            return_other=global_step % log_step == 0,
        )

        if global_step % log_step == 0:
            for key, value in other_res.items():
                if isinstance(value, torch.Tensor):
                    value = float(value.mean().cpu().item())
                records[key] = value

        return loss, losses, records

    # interface
    def on_before_zero_grad(self):
        pass

    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for key, values in records.items():
            if isinstance(values, torch.Tensor) and len(values.shape) > 1:
                logger.add_image(f"{prefix}{key}", values, global_step=global_step)
            elif isinstance(values, float):
                logger.add_scalar(f"{prefix}{key}", values, global_step=global_step)


class MultiDistillerForPretrain(nn.Module):
    """
    Distiller for pretraining with flexible number of teacher models.
    """

    def __init__(self, datarc: edict, config: MultiDistillerConfig, teacher_config: edict):
        super().__init__()
        self.config = config
        self.datarc = datarc
        self.distiller = MultiDistillerModel(config)
        #print(f"the distiller model arch inside MultiDistillerForPretrain is {self.distiller}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.teacher_config = teacher_config
        print(f"the teacher config inside MultiDistillerForPretrain is {self.teacher_config}")
        self.teachers = teacher_config.models  # Expecting a list of teacher model names
        self.train_on_all_data = self.config.train_on_all_data
        # Dictionary to store teacher models and processors
        self.teacher_models = {}
        self.teacher_processors = {}
        self.last_loss = []
        
        # Load teacher models based on self.teachers
        for model_name in self.teachers:
            if model_name == 'hubert_base':
                teacher_1 = torch.hub.load("s3prl/s3prl",model_name).to(device)
                if model_name.find("hubert") >= 0 or model_name.find("wav2vec2") >= 0:
                    teacher_1.model.encoder.layerdrop = 0
                    print("[HuBERT] - Disabled teacher's encoder layerdrop")
                self.teacher_models[model_name] = teacher_1
                print(self.teacher_models[model_name])
            elif model_name == 'mert_v0_public':
                temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
                temp_config.output_hidden_states = True  # Enable hidden states in the output
                teacher_2 = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).to(device)
                disable_MERT_encoder_dropout(teacher_2)
                self.teacher_models[model_name] = teacher_2
            elif model_name == 'ssast_frame':
                self.temporal_alignment = TemporalAligner()
                teacher_3 = ASTModel(fshape=128, tshape=2, fstride=128, tstride=1, input_tdim=1024, input_fdim=128,
                                  model_size='base', pretrain_stage=False, load_pretrained_mdl_path="/home/twsgxyc199/ycevan/mdd/dataset/SSAST-Base-Frame-400.pth").to(device)
                teacher_3_processor = FeatureExtractor(target_length=1024, apply_cmvn=False)
                print(f"teacher_3_processor is {teacher_3_processor}")
                disable_SSAST_encoder_dropout(teacher_3)
                self.teacher_models[model_name] = teacher_3
                self.teacher_processors[model_name] = teacher_3_processor
            elif model_name == 'whisper_small':
                self.temporal_alignment = TemporalAligner()
                from transformers import WhisperModel, WhisperProcessor
                # Load the Whisper model and processor
                print("[Whisper] - Loading Whisper small model")
                teacher_4 = WhisperModel.from_pretrained("openai/whisper-small").to(device)
                teacher_4.config.output_hidden_states = True  # Enable hidden states in the output
                teacher_4_processor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
                self.teacher_processors[model_name] = teacher_4_processor
                print("[Whisper] - Enabled output_hidden_states")
                # Create a processor for preprocessing Whisper inputs
                print(f"teacher_4_processor is {teacher_4_processor}")
                
                # Add the Whisper model and processor to the teacher models and processors
                self.teacher_models[model_name] = teacher_4
                self.teacher_processors[model_name] = teacher_4_processor
            elif model_name == 'wavlm':
                print("[WavLM] - Loading WavLM model")
                teacher_5_processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
                self.teacher_processors[model_name] = teacher_5_processor
                self.teacher_models[model_name] = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus").to(device)
                disable_wavlm_encoder_dropout(self.teacher_models[model_name]) 
                print(self.teacher_models[model_name])
            elif model_name == 'emotion2vec':
                print("[emotion2vec] - Loading emotion2vec model")
                model_path = UserDirModule('/home/twsgxyc199/ycevan/mdd/s3prl/s3prl/emotion2vec/emo_upstream')
                fairseq.utils.import_user_module(model_path)
                model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['/home/twsgxyc199/ycevan/mdd/dataset/emotion2vec_base.pt'])
                model = model[0]
                model.eval()
                model.to(self.device)
                self.teacher_models[model_name] = model
            else:
                print(f"Warning: Unknown teacher model {model_name} specified.")
            
        
        # Freeze all teacher models
        for teacher in self.teacher_models.values():
            freeze_model(teacher)
        
        # Initialize loss function
        if config.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="none")
        elif config.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError(config.loss_type)

        self.cosine_loss = config.cosine_loss
        if self.cosine_loss > 0:
            print("[DistillerForPretrain] - Enabled cosine similarity loss.")
        
        # Ensure that we can only load weights from hubert_base or mert_v0_public
        model_to_initialize = self.config.initialize_from[0]
        if model_to_initialize == 'ast':
            raise AssertionError("[Error] Cannot initialize weights from 'ast' model. The student's architecture is compatible only with 'hubert_base' or 'mert_v0_public'.")
        elif model_to_initialize == 'hubert_base':
            print(f"Initializing student model from {model_to_initialize}...")
            self.load_teacher_weights('hubert_base')
        elif model_to_initialize == 'mert_v0_public':
            print(f"Initializing student model from {model_to_initialize}...")
            self.load_teacher_weights('mert_v0_public')
        elif model_to_initialize == 'avg':
            print(f"Initializing student model from {model_to_initialize}...")
            self.load_teacher_weights('avg')
        elif model_to_initialize == 'wavlm':
            print(f"Initializing student model from {model_to_initialize}...")
            self.load_teacher_weights('wavlm')
        elif model_to_initialize == 'whisper_small':
            print(f"Initializing student model from {model_to_initialize}...")
            self.load_teacher_weights('whisper_small')
        elif model_to_initialize == 'SVD':
            print(f"Initializing student model from {model_to_initialize}...")
            self.load_teacher_weights('SVD')

    def load_teacher_weights(self, teacher_name, device="cuda"):
        """
        Load the weights from a specified teacher model (hubert_base or mert_v0_public).
        """
        teacher_model = self.teacher_models.get(teacher_name)
        if teacher_model is None:
            print(f"teacher_name is {teacher_name} and self.config.initialize_from is {self.config.initialize_from[0]} ")
            if teacher_name == self.config.initialize_from[0]:
                if teacher_name == "mert_v0_public":
                    temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
                    temp_config.output_hidden_states = True  # Enable hidden states in the output
                    teacher_model = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).to(device)
                    disable_MERT_encoder_dropout(teacher_model)
                if teacher_name == "hubert_base":
                    teacher_model = torch.hub.load("s3prl/s3prl","hubert_base").to(device)
                    teacher_model.model.encoder.layerdrop = 0
                    print("[HuBERT] - Disabled teacher's encoder layerdrop")
                if teacher_name == 'wavlm':
                    teacher_model = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus").to(device)
                elif teacher_name == "SVD":
                    print("fuck")
            else:
                raise ValueError(f"[Error] Teacher model '{teacher_name}' not found in the loaded teacher models.")
        if teacher_name == 'wavlm':
            print(f"[DistillerForPretrain] - Loading weights from {teacher_name}")
    
            # Load weights for feature extractor
            if self.config.init_teacher_conv_layers:
                print(f"[DistillerForPretrain] - Initializing feature extractor from {teacher_name}")
                wavlm_state_dict = teacher_model.feature_extractor.state_dict()
                mapped_state_dict = {}
                for key, value in wavlm_state_dict.items():
                    if "conv.weight" in key:
                        # Map "conv_layers.X.conv.weight" -> "conv_layers.X.0.weight"
                        new_key = key.replace(".conv.weight", ".0.weight")
                    elif "layer_norm.weight" in key:
                        # Map "conv_layers.X.layer_norm.weight" -> "conv_layers.X.2.weight"
                        new_key = key.replace("layer_norm.weight", "2.weight")
                    elif "layer_norm.bias" in key:
                        # Map "conv_layers.X.layer_norm.bias" -> "conv_layers.X.2.bias"
                        new_key = key.replace("layer_norm.bias", "2.bias")
                    else:
                        new_key = key
                    mapped_state_dict[new_key] = value
                self.distiller.feature_extractor.load_state_dict(
                    mapped_state_dict
                )
                if self.distiller.post_extract_proj is not None:
                    self.distiller.post_extract_proj.load_state_dict(
                        teacher_model.feature_projection.projection.state_dict()
                    )
    
            # Load weights for encoder layers
            if self.config.init_teacher_encoder_layers:
                print(f"[DistillerForPretrain] - Initializing encoder from {teacher_name}")
                pos_conv_state_dict = teacher_model.encoder.pos_conv_embed.conv.state_dict()
                original_weight = pos_conv_state_dict["parametrizations.weight.original0"]
                weight_g = original_weight.norm(dim=1, keepdim=True)
                weight_v = original_weight / weight_g
                mapped_pos_conv_state_dict = {
                    "0.weight": original_weight,
                    #"0.weight_v": weight_v,
                    "0.bias": pos_conv_state_dict["bias"],
                }
                # Load positional convolution embeddings
                self.distiller.encoder.pos_conv.load_state_dict(mapped_pos_conv_state_dict, strict=False)
        
                # Load encoder layer weights
                for l in range(self.config.encoder_layers):
                    print(teacher_model.encoder.layers[l].state_dict().keys())
                    for key, param in teacher_model.encoder.layers[l].state_dict().items():
                    
                        print(f"  {key}: {param.shape}")
                    wavlm_layer_state_dict = teacher_model.encoder.layers[l].state_dict()
                    mapped_layer_state_dict = {}
                    for key, value in wavlm_layer_state_dict.items():
                        if key.startswith("attention.k_proj"):
                            mapped_key = key.replace("attention.", "self_attn.").replace("k_proj", "k_proj")
                        elif key.startswith("attention.v_proj"):
                            mapped_key = key.replace("attention.", "self_attn.").replace("v_proj", "v_proj")
                        elif key.startswith("attention.q_proj"):
                            mapped_key = key.replace("attention.", "self_attn.").replace("q_proj", "q_proj")
                        elif key.startswith("attention.out_proj"):
                            mapped_key = key.replace("attention.", "self_attn.").replace("out_proj", "out_proj")
                        elif key.startswith("layer_norm"):
                            mapped_key = key.replace("layer_norm", "self_attn_layer_norm")
                        elif key.startswith("feed_forward.intermediate_dense"):
                            mapped_key = key.replace("feed_forward.intermediate_dense", "fc1")
                        elif key.startswith("feed_forward.output_dense"):
                            mapped_key = key.replace("feed_forward.output_dense", "fc2")
                        elif key.startswith("final_layer_norm"):
                            # Map final_layer_norm keys
                            mapped_key = key
                        else:
                            # Skip keys that don't map to the distiller
                            print(f"Skipping WavLM key: {key}")
                            continue
                        mapped_layer_state_dict[mapped_key] = value
                    self.distiller.encoder.layers[l].load_state_dict(mapped_layer_state_dict, strict=False)
    
                print(f"[DistillerForPretrain] - WavLM initialization completed!")            
        if teacher_name == 'avg':
            print(f"[DistillerForPretrain] - Loading weights from {teacher_name}")
            
            # Load weights for feature extractor
            if self.config.init_teacher_conv_layers:
                print(f"[DistillerForPretrain] - Initializing feature extractor from {teacher_name}")
                self.distiller.feature_extractor.load_state_dict(
                    teacher_model.model.feature_extractor.state_dict()
                )
                if self.distiller.post_extract_proj is not None:
                    self.distiller.post_extract_proj.load_state_dict(
                        teacher_model.model.post_extract_proj.state_dict()
                    )
            
            # Load weights for encoder layers
            if self.config.init_teacher_encoder_layers:
                print(f"[DistillerForPretrain] - Initializing encoder from {teacher_name}")
                self.distiller.encoder.pos_conv.load_state_dict(
                    teacher_model.model.encoder.pos_conv.state_dict()
                )
                for l in range(self.config.encoder_layers):
                    converted_state_dict_mert = rename_attention_keys_mert(self.teacher_models['mert_v0_public'].encoder.layers[l].state_dict())
                    state_dict_hubert = self.teacher_models['hubert_base'].model.encoder.layers[l].state_dict()
                    averaged_encoder = average_weights([converted_state_dict_mert, state_dict_hubert])
                    student_encoder = self.distiller.encoder.layers[l].state_dict()
                    for k, v in averaged_encoder.items():
                        if k in student_encoder:
                            student_encoder[k] = v
                    self.distiller.encoder.layers[l].load_state_dict(
                        student_encoder
                    )
        # Example: loading weights from hubert_base or mert_v0_public for feature extractor
        if teacher_name == 'hubert_base':
            print(f"[DistillerForPretrain] - Loading weights from {teacher_name}")
            
            # Load weights for feature extractor
            if self.config.init_teacher_conv_layers:
                print(f"[DistillerForPretrain] - Initializing feature extractor from {teacher_name}")
                self.distiller.feature_extractor.load_state_dict(
                    teacher_model.model.feature_extractor.state_dict()
                )
                if self.distiller.post_extract_proj is not None:
                    self.distiller.post_extract_proj.load_state_dict(
                        teacher_model.model.post_extract_proj.state_dict()
                    )
            
            # Load weights for encoder layers
            if self.config.init_teacher_encoder_layers:
                print(f"[DistillerForPretrain] - Initializing encoder from {teacher_name}")
                self.distiller.encoder.pos_conv.load_state_dict(
                    teacher_model.model.encoder.pos_conv.state_dict()
                )
                for l in range(self.config.encoder_layers):
                    print(teacher_model.model.encoder.layers[l].state_dict().keys())
                    for key, param in teacher_model.model.encoder.layers[l].state_dict().items():
                        print(f"  {key}: {param.shape}")
                    self.distiller.encoder.layers[l].load_state_dict(
                        teacher_model.model.encoder.layers[l].state_dict()
                    )

        if teacher_name == 'mert_v0_public':
            print(f"[DistillerForPretrain] - Loading weights from {teacher_name}")
            # Load weights for feature extractor
            # Retrieve the state_dict of the MERT feature extractor
            state_dict = teacher_model.feature_extractor.state_dict()

            # Modify the keys to match distilHuBERT's expected layer names
            new_state_dict = {}
            for key, value in state_dict.items():
                # Convert "conv_layers.0.conv.weight" to "conv_layers.0.0.weight"
                # Convert "conv_layers.0.layer_norm.weight" to "conv_layers.0.2.weight" (assuming layer_norm is at index 2)
                if "conv_layers" in key:
                    # Handle the convolution layers
                    if "conv.weight" in key:
                        new_key = key.replace("conv.weight", "0.weight")
                    # Handle the normalization layers
                    elif "layer_norm" in key:
                        new_key = key.replace("layer_norm.weight", "2.weight").replace("layer_norm.bias", "2.bias")
                    # Handle activation layers if needed (you can add this if distilHuBERT expects it)
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            if self.config.init_teacher_conv_layers:
                print(f"[DistillerForPretrain] - Initializing feature extractor from {teacher_name}")
                self.distiller.feature_extractor.load_state_dict(new_state_dict)

                self.distiller.post_extract_proj.load_state_dict(
                teacher_model.feature_projection.projection.state_dict()
                )
            

            # Load weights for encoder layers
            if self.config.init_teacher_encoder_layers:
                # MERT has `conv`, `padding`, `activation`, distilHuBERT has indices `0`, `1`, `2` in a Sequential
                mert_pos_conv = teacher_model.encoder.pos_conv_embed.state_dict()

                # Decompose the weight_g and weight_v to get the actual weights
                #conv_weight_g = mert_pos_conv['conv.weight_g']
                #conv_weight_v = mert_pos_conv['conv.weight_v']
                
                #conv_weight = (conv_weight_g / conv_weight_v.norm(dim=1, keepdim=True)) * conv_weight_v # ->
                # -> check: https://pytorch.org/docs/2.3/generated/torch.nn.utils.weight_norm.html

                # Create a new state_dict for distilHuBERT by mapping the keys
                # Create a new state_dict to map MERT's keys to distilHuBERT's keys
                pos_conv_dict = {
                    '0.bias': mert_pos_conv['conv.bias'],            # Mapping MERT's conv.bias to 0.bias
                    '0.weight_g': mert_pos_conv['conv.parametrizations.weight.original0'],    # Mapping MERT's weight_g to 0.weight_g
                    '0.weight_v': mert_pos_conv['conv.parametrizations.weight.original1']    # Mapping MERT's weight_v to 0.weight_v
                }
        

                print(f"[DistillerForPretrain] - Loading encoder positional convolution from MERT")
                self.distiller.encoder.pos_conv.load_state_dict(pos_conv_dict)
                
                print(f"[DistillerForPretrain] - Loading encoder layers from MERT")
                for l in range(self.config.encoder_layers):
                    # Mapping MERT's HubertEncoderLayer to distilHuBERT's TransformerSentenceEncoderLayer
                    mert_encoder_layer = teacher_model.encoder.layers[l].state_dict()
                    # Create a new state dict with mapped keys for distilHuBERT
                    new_encoder_layer_dict = {}
                    for key, value in mert_encoder_layer.items():
                        # Rename attention block
                        if 'attention.' in key:
                            new_key = key.replace('attention.', 'self_attn.')
                        # Rename layer_norm to self_attn_layer_norm
                        elif 'layer_norm' in key and 'final_layer_norm' not in key:
                            new_key = key.replace('layer_norm', 'self_attn_layer_norm')
                        elif 'final_layer_norm' in key:
                            new_key = key  # No changes for final_layer_norm
                        # Rename feed forward layers
                        elif 'feed_forward.intermediate_dense' in key:
                            new_key = key.replace('feed_forward.intermediate_dense', 'fc1')
                        elif 'feed_forward.output_dense' in key:
                            new_key = key.replace('feed_forward.output_dense', 'fc2')
                        else:
                            new_key = key  # If no changes are needed, keep the key the same
                        # Add the mapped key and value to the new dict
                        new_encoder_layer_dict[new_key] = value

                    self.distiller.encoder.layers[l].load_state_dict(new_encoder_layer_dict)

        if teacher_name == 'whisper_small':
            print(f"[DistillerForPretrain] - Loading weights from {teacher_name}")
            print(teacher_model)
            print(teacher_model.encoder)

            # Load weights for encoder's convolutional layers
            if self.config.init_teacher_conv_layers:
                print(f"[DistillerForPretrain] - Initializing encoder's convolutional layers from {teacher_name}")
                # Map weights and biases for conv1
                self.distiller.encoder.conv1.weight.data.copy_(
                    teacher_model.encoder.conv1.weight.data
                )
                self.distiller.encoder.conv1.bias.data.copy_(
                    teacher_model.encoder.conv1.bias.data
                )

                # Map weights and biases for conv2
                self.distiller.encoder.conv2.weight.data.copy_(
                    teacher_model.encoder.conv2.weight.data
                )
                self.distiller.encoder.conv2.bias.data.copy_(
                    teacher_model.encoder.conv2.bias.data
                )

            # Load weights for encoder transformer layers
            if self.config.init_teacher_encoder_layers:
                print(f"[DistillerForPretrain] - Initializing encoder transformer layers from {teacher_name}")
                for l in range(min(self.config.encoder_layers, len(teacher_model.encoder.layers))):
                    self.distiller.encoder.layers[l].load_state_dict(
                        teacher_model.encoder.layers[l].state_dict()
                    )
            
            # Load weights for decoder layers (optional, if needed for distillation)
            if hasattr(self.distiller, "decoder") and self.config.init_teacher_decoder_layers:
                print(f"[DistillerForPretrain] - Initializing decoder transformer layers from {teacher_name}")
                for l in range(min(self.config.decoder_layers, len(teacher_model.decoder.layers))):
                    self.distiller.decoder.layers[l].load_state_dict(
                        teacher_model.decoder.layers[l].state_dict()
                    )

        if teacher_name == 'SVD':
            print(f"[DistillerForPretrain] - Loading weights Using SVD")
            temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
            temp_config.output_hidden_states = True  # Enable hidden states in the output
            mert_teacher = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).to(device)
            disable_MERT_encoder_dropout(mert_teacher)
            hubert_teacher = torch.hub.load("s3prl/s3prl","hubert_base").to(device)
            hubert_teacher.model.encoder.layerdrop = 0
            # Load weights for feature extractor
            if self.config.init_teacher_conv_layers:
                print(f"[DistillerForPretrain] - Initializing feature extractor from {teacher_name}")                
                self.distiller.feature_extractor.load_state_dict(
                    hubert_teacher.model.feature_extractor.state_dict()
                )
                if self.distiller.post_extract_proj is not None:
                    self.distiller.post_extract_proj.load_state_dict(
                        hubert_teacher.model.post_extract_proj.state_dict()
                    )
            
            # Load weights for encoder layers
            if self.config.init_teacher_encoder_layers:
                print(f"[DistillerForPretrain] - Initializing encoder from {teacher_name}")
                self.distiller.encoder.pos_conv.load_state_dict(
                    hubert_teacher.model.encoder.pos_conv.state_dict()
                )
                for l in range(self.config.encoder_layers):
                    hubert_layer = hubert_teacher.model.encoder.layers[l]
                    mert_layer = mert_teacher.encoder.layers[l]
                    print(f"Initializing Layer {l} with SVD (HuBERT + MERT)")
                    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                        print(f"  Processing {proj_name}")
                        #print(hubert_layer,'dd', mert_layer)
                        # HuBERT weights
                        hubert_weight = getattr(hubert_layer.self_attn, proj_name).weight.data
                        hubert_bias = getattr(hubert_layer.self_attn, proj_name).bias.data

                        # MERT weights
                        mert_weight = getattr(mert_layer.attention, proj_name).weight.data
                        mert_bias = getattr(mert_layer.attention, proj_name).bias.data
                        print('original shape: ', mert_weight.shape, mert_bias.shape)
                        # Perform SVD on HuBERT weight
                        U_hubert, S_hubert, Vh_hubert = torch.linalg.svd(hubert_weight, full_matrices=False)
                        hubert_truncated = (U_hubert[:, :384] @ torch.diag(S_hubert[:384])) @ Vh_hubert[:384, :]

                        # Perform SVD on MERT weight
                        U_mert, S_mert, Vh_mert = torch.linalg.svd(mert_weight, full_matrices=False)
                        mert_truncated = (U_mert[:, :384] @ torch.diag(S_mert[:384])) @ Vh_mert[:384, :]

                        # Concatenate HuBERT and MERT results
                        concatenated_weight = torch.cat((hubert_truncated, mert_truncated), dim=0)
                        
                        # Load into student model
                        getattr(self.distiller.encoder.layers[l].self_attn, proj_name).weight.data = concatenated_weight
                        concatenated_bias = torch.cat((hubert_bias[:384], mert_bias[:384]))
                        getattr(self.distiller.encoder.layers[l].self_attn, proj_name).bias.data = concatenated_bias
                        print(proj_name, 'concated shape: ', concatenated_weight.shape, concatenated_bias.shape)
                    # Initialize feedforward layers (fc1, fc2)
                    for linear_name, output_dim in [('fc1', 3072), ('fc2', 768)]:
                        print(f"  Processing {linear_name}")
                        
                        # HuBERT weights
                        hubert_weight = getattr(hubert_layer, linear_name).weight.data
                        hubert_bias = getattr(hubert_layer, linear_name).bias.data
                        print(linear_name, 'origin', hubert_weight.shape, hubert_bias.shape)
                        # MERT weights
                        if linear_name=='fc1':
                            linear_name_mert = 'intermediate_dense'
                        else:
                            linear_name_mert = 'output_dense'
                        # print(mert_layer.feed_forward)
                        mert_weight = getattr(mert_layer.feed_forward, linear_name_mert).weight.data
                        mert_bias = getattr(mert_layer.feed_forward, linear_name_mert).bias.data

                        # Perform SVD on HuBERT weight
                        U_hubert, S_hubert, Vh_hubert = torch.linalg.svd(hubert_weight, full_matrices=False)
                        if linear_name == 'fc1':
                            hubert_truncated = (U_hubert[:, :1536] @ torch.diag(S_hubert[:1536])) @ Vh_hubert[:1536, :]
                        else:  # fc2
                            hubert_truncated = (U_hubert[:, :384] @ torch.diag(S_hubert[:384])) @ Vh_hubert[:384, :]

                        # Perform SVD on MERT weight
                        U_mert, S_mert, Vh_mert = torch.linalg.svd(mert_weight, full_matrices=False)
                        if linear_name == 'fc1':
                            mert_truncated = (U_mert[:, :1536] @ torch.diag(S_mert[:1536])) @ Vh_mert[:1536, :]
                        else:  # fc2
                            mert_truncated = (U_mert[:, :384] @ torch.diag(S_mert[:384])) @ Vh_mert[:384, :]

                        # Concatenate HuBERT and MERT results
                        concatenated_weight = torch.cat((hubert_truncated, mert_truncated), dim=0)

                        # Load into student model
                        getattr(self.distiller.encoder.layers[l], linear_name).weight.data = concatenated_weight
                        if linear_name == 'fc1':
                            concatenated_bias = torch.cat((hubert_bias[:1536], mert_bias[:1536]))
                        else:
                            concatenated_bias = torch.cat((hubert_bias[:384], mert_bias[:384]))
                        getattr(self.distiller.encoder.layers[l], linear_name).bias.data = concatenated_bias
                        print(linear_name, 'concated shape: ', concatenated_weight.shape, concatenated_bias.shape)
                    # Initialize layer normalization weights and biases
                    for norm_name in ['self_attn_layer_norm', 'final_layer_norm']:
                        print(f"  Processing {norm_name}")

                        # HuBERT normalization weights and biases
                        hubert_weight = getattr(hubert_layer, norm_name).weight.data
                        hubert_bias = getattr(hubert_layer, norm_name).bias.data

                        # MERT normalization weights and biases
                        if norm_name=='self_attn_layer_norm':
                            norm_name_mert = 'layer_norm'
                        mert_weight = getattr(mert_layer, norm_name_mert).weight.data
                        mert_bias = getattr(mert_layer, norm_name_mert).bias.data

                        # Concatenate and load
                        concatenated_weight = torch.cat((hubert_weight[:384], mert_weight[:384]))
                        concatenated_bias = torch.cat((hubert_bias[:384], mert_bias[:384]))
                        getattr(self.distiller.encoder.layers[l], norm_name).weight.data = concatenated_weight
                        getattr(self.distiller.encoder.layers[l], norm_name).bias.data = concatenated_bias

                    print(f"Layer {l} initialized with SVD (HuBERT + MERT).")



    def forward(
        self,
        wave_input: torch.Tensor,
        wave_orig_16k: list,
        sample_domain: list,
        wave_len: torch.Tensor,
        pad_mask: torch.Tensor,
        return_other: bool = False,
    ):
        """
        Forward function.
        """
        feat, feat_final, pred, pad_mask = self.distiller(wave_input, pad_mask)
        teachers_hidden_states = {}
        with torch.no_grad():
            wave_orig_16k = [wave.to(wave_input.device) for wave in wave_orig_16k]
            if isinstance(wave_orig_16k, list):
                    max_length = max(wave.size(0) for wave in wave_orig_16k)
                    padded_wave_orig = [F.pad(wave, (0, max_length - wave.size(0))) for wave in wave_orig_16k]
                    wave_orig_16k = torch.stack(padded_wave_orig).to(wave_input.device)
            # wave_orig_24k = [wave.to(wave_input.device) for wave in wave_orig_24k]
            # if isinstance(wave_orig_24k, list):
            #         max_length = max(wave.size(0) for wave in wave_orig_24k)
            #         padded_wave_orig = [F.pad(wave, (0, max_length - wave.size(0))) for wave in wave_orig_24k]
            #         wave_orig_24k = torch.stack(padded_wave_orig).to(wave_input.device)
            with torch.cuda.amp.autocast(False):
                # Loop through the teacher models to gather hidden states
                for model_name, teacher in self.teacher_models.items():
                    if model_name == 'hubert_base':
                        teacher_hiddens = teacher(wave_orig_16k)
                    elif model_name == 'mert_v0_public':
                        teacher_hiddens = teacher(wave_orig_16k)
                    elif model_name == 'ssast_frame':
                        features = [self.teacher_processors[model_name](wav.unsqueeze(0)) for wav in wave_orig_16k]
                        features = torch.stack(features, dim=0)
                        teacher_hiddens, features = teacher(features)
                        teacher_hiddens = torch.stack(teacher_hiddens)
                        padded_hidden_states = F.pad(teacher_hiddens, (0, 0, 0, 0, 0, 0, 1, 0)) # Adds one dimension from 12 to 13 at the start
                        teacher_hiddens = {"hidden_states": padded_hidden_states}
                    elif model_name == 'whisper_small':
                        # Process waveform with Whisper processor if necessary
                        processor = self.teacher_processors[model_name]
                        features = processor([wav.cpu().numpy() for wav in wave_orig_16k], sampling_rate=16000, return_tensors="pt").input_features
                        features = features.to(self.device)
                        # Forward pass through the Whisper model
                        decoder_input_ids = torch.tensor([[1, 1]]) * teacher.config.decoder_start_token_id
                        decoder_input_ids = decoder_input_ids.to(self.device)

                        outputs = teacher(features, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                        teacher_hiddens = {"hidden_states": outputs.encoder_hidden_states}
                                    # Extract hidden states based on task embedding type
                    elif model_name == 'wavlm':
                        processor = self.teacher_processors[model_name]
                        features = processor([wav.cpu().numpy() for wav in wave_orig_16k], sampling_rate=16000, return_tensors="pt")
                        features = features.to(self.device)
                        outputs = teacher(**features, output_hidden_states=True)
                        teacher_hiddens = {"hidden_states": outputs.hidden_states}
                    elif model_name == 'emotion2vec':
                        
                        source = wave_orig_16k.float().to(self.device)
                        #print(f"Original source shape: {source.shape}")
                        #source = source.view(1, -1)
                        padding_mask = (source != 0).unsqueeze(1).unsqueeze(2)
                        feats = teacher.extract_features(source, padding_mask=padding_mask)
                        teacher_hiddens = {"hidden_states": feats["layer_results"]}
                        #print(feats["layer_results"][0].shape)

                    if self.config.task_emb_type in ["expand-last", "hnet", "self-hidden"]:
                        # print(model_name, teacher_hiddens["hidden_states"][0].shape)
                        if model_name == 'emotion2vec':
                            teacher_hiddens = [
                                teacher_hiddens["hidden_states"][i]
                                for i in [1,4,7]
                            ]
                        else:
                            teacher_hiddens = [
                                teacher_hiddens["hidden_states"][i]
                                for i in self.distiller.pred_layer_id
                            ]
                        teachers_hidden_states[model_name] = torch.stack(teacher_hiddens, dim=1)
                        #print(model_name, teachers_hidden_states[model_name].shape)

        # Compute all objectives
        (
            total_loss,
            total_losses,
            rec_loss,
            rec_layer_loss_dict,
            feat_pen,
            sim_loss,
            sim_layer_loss_dict,
        ) = self.compute_loss(feat, pred, teachers_hidden_states, sample_domain, return_other)

        if return_other:
            with torch.no_grad():
                other_res = {
                    "rec_loss": rec_loss,
                    "feat_pen": feat_pen,
                    "sim_loss": sim_loss,
                    "norm_feat_final": feat_final.pow(2).mean(),
                }

                # Initialize a dictionary to keep norms for each teacher
                teacher_norms = {}

                # Calculate norms for each teacher and add to teacher_norms
                for model_name, hidden_states in teachers_hidden_states.items():
                    teacher_norms[model_name] = torch.abs(hidden_states).mean((0, 2, 3))

                # Log metrics for each teacher
                for model_name, norm in teacher_norms.items():
                    # Retrieve the layer-wise losses from the dictionaries
                    rec_layer_loss = rec_layer_loss_dict.get(model_name, None)
                    sim_layer_loss = sim_layer_loss_dict.get(model_name, None)

                    if rec_layer_loss is not None:
                        # If task_emb_type is 'none', log only the first layer as before
                        if self.config.task_emb_type == "none":
                            other_res[f"rec_l_{model_name}_{self.config.n_tasks}"] = rec_layer_loss[0]
                            other_res[f"tar_norm_l_{model_name}_{self.config.n_tasks}"] = norm[0]
                            if sim_layer_loss is not None:
                                other_res[f"sim_l_{model_name}_{self.config.n_tasks}"] = sim_layer_loss[0]
                        else:
                            # Otherwise, log all layers or based on pred_layer_id
                            for i in range(min(self.config.n_tasks, len(rec_layer_loss))):
                                layer_id = i + 1
                                if self.config.task_emb_type in ["expand-last", "hnet", "self-hidden"]:
                                    layer_id = self.distiller.pred_layer_id[i]

                                # Logging for each layer of each teacher
                                other_res[f"rec_l_{model_name}_{layer_id}"] = rec_layer_loss[i]
                                other_res[f"tar_norm_l_{model_name}_{layer_id}"] = norm[i]
                                if sim_layer_loss is not None and i < len(sim_layer_loss):
                                    other_res[f"sim_l_{model_name}_{layer_id}"] = sim_layer_loss[i]

                # Additional task embedding logging if applicable
                if self.config.task_emb_type not in ["expand-last", "hnet", "self-hidden"]:
                    other_res["norm_task_emb"] = self.distiller.task_embedding.weight.pow(2).mean()
        else:
            other_res = None



        return total_loss, total_losses, other_res

    def l2_normalize(self, tensor):
        return tensor / tensor.norm(p=2, dim=-1, keepdim=True)

    # def compute_loss(self, feat, pred, target, sample_domain, return_other=False):
    #     """
    #     Computes loss for multiple teachers.
    #     Inputs:
    #         feat: B x T x D
    #         pred: Dict containing predictions from multiple teachers
    #         target: Dict containing targets corresponding to each teacher
    #         return_other: Flag to indicate if additional losses should be returned
    #     """
    #     # Initialize variables to accumulate losses
    #     total_loss = 0
    #     total_rec_loss = 0
    #     total_sim_loss = 0
    #     total_feat_pen = 0

    #     rec_layer_loss_dict = {}
    #     sim_layer_loss_dict = {}

    #     #print(f"fix here for when you use more teachers....")

    #     # Iterate over each teacher's predictions and targets
    #     for teacher_key in target.keys(): ## on the meantime.... this needs to be fixed
    #         # teacher_pred = pred    # [teacher_key]  # Prediction from the current teacher
    #         teacher_pred = pred[teacher_key]
    #         teacher_target = target[teacher_key]  # Target corresponding to the current teacher

    #         if 'ssast_frame' in self.teacher_models:
    #             aligned_preds = []  # To store aligned student features
    #             aligned_targets = []  # To store aligned teacher features

    #             for i in range(teacher_pred.shape[1]): ### do this outside... is better and more efficient, capitalize one of the for already being done outside...
    #                 align_teacher, align_student = self.temporal_alignment(teacher_target[:,i,:,:], teacher_pred[:,i,:,:])
    #                 # Append the aligned features to the lists
    #                 aligned_preds.append(align_student.unsqueeze(1))  # Add back the layer dimension
    #                 aligned_targets.append(align_teacher.unsqueeze(1))  # Add back the layer dimension

    #             # Concatenate aligned layers back to 4D tensors (batch, layers, time, feature_dim)
    #             teacher_pred = torch.cat(aligned_preds, dim=1)
    #             teacher_target = torch.cat(aligned_targets, dim=1)
            
    #         # teacher_pred = self.l2_normalize(teacher_pred)
    #         # teacher_target = self.l2_normalize(teacher_target)
    #         # # Ensure shapes match
    #         # print(teacher_key, ':', teacher_pred.shape, ' ', teacher_target.shape)
    #         assert teacher_pred.shape == teacher_target.shape, (teacher_pred.shape, teacher_target.shape)

    #         if self.train_on_all_data:
    #             rec_loss = self.loss_func(teacher_pred, teacher_target)  # B x N x T x D
    #             if teacher_key == 'ssast_frame':
    #                 weighted_loss = rec_loss.mean() * 0.05
    #             else:
    #                 weighted_loss = rec_loss.mean()

    #             total_rec_loss += weighted_loss
    #         else:
    #             temp_shape = (1, *teacher_target.shape[1:])
    #             teacher_pred_for_spec_domain = []
    #             teacher_target_for_spec_domain = []

    #             for i, domain in enumerate(sample_domain):
    #                 if domain == teacher_key:
    #                     # Select the corresponding prediction and target
    #                     teacher_pred_for_spec_domain.append(teacher_pred[i])  # B x N x T x D for sample i
    #                     teacher_target_for_spec_domain.append(teacher_target[i])
                
    #             if len(teacher_pred_for_spec_domain) > 0:
    #                 teacher_pred = torch.stack(teacher_pred_for_spec_domain)
    #                 teacher_target = torch.stack(teacher_target_for_spec_domain)
    #                 # Compute reconstruction loss
    #                 rec_loss = self.loss_func(teacher_pred, teacher_target)  # B x N x T x D
    #                 if teacher_key == 'ssast_frame':
    #                     weighted_loss = rec_loss.mean() * 0.05
    #                 else:
    #                     weighted_loss = rec_loss.mean()
    #             else:
    #                 print(f"No matching predictions found for teacher: {teacher_key}")
    #                 weighted_loss = 0
    #                 rec_loss = torch.zeros(temp_shape)
    #             total_rec_loss += weighted_loss

    #         # Optionally compute layer-wise reconstruction loss
    #         if return_other:
    #             with torch.no_grad():
    #                 if isinstance(rec_loss, torch.Tensor) and rec_loss.numel() > 0:
    #                     # If it's a tensor with elements, calculate the mean loss
    #                     rec_layer_loss = rec_loss.mean((0, 2, 3))
    #                 if teacher_key == 'ssast_frame':
    #                     rec_layer_loss = rec_layer_loss * 0.05
    #                 else:
    #                     rec_layer_loss = rec_layer_loss
    #             rec_layer_loss_dict[teacher_key] = rec_layer_loss
    #         else:
    #             rec_layer_loss_dict[teacher_key] = None

    #         # Compute cosine similarity loss if applicable
    #         if self.cosine_loss > 0:
    #             sim_loss = -F.logsigmoid(F.cosine_similarity(teacher_pred, teacher_target, dim=-1))  # B x N x T
    #             total_sim_loss += sim_loss.mean()

    #             # Optionally compute layer-wise similarity loss
    #             if return_other:
    #                 with torch.no_grad():
    #                     sim_layer_loss = sim_loss.mean((0, 2))  # Per-layer similarity loss
    #                 sim_layer_loss_dict[teacher_key] = sim_layer_loss
    #             else:
    #                 sim_layer_loss_dict[teacher_key] = None
    #         else:
    #             sim_layer_loss = 0
    #             sim_layer_loss_dict[teacher_key] = None

    #         # Compute feature penalty loss
    #         feat_pen = feat.float().pow(2).mean()
    #         total_feat_pen += feat_pen

    #     # Sum up the total loss components
    #     total_loss = (
    #         total_rec_loss
    #         + total_feat_pen * self.config.feat_pen_loss
    #         + total_sim_loss * self.cosine_loss
    #     )

    #     return total_loss, total_rec_loss, rec_layer_loss_dict, total_feat_pen, total_sim_loss, sim_layer_loss_dict
    def compute_loss(self, feat, pred, target, sample_domain, return_other=False):
        """
        Computes loss for multiple teachers.
        Inputs:
            feat: B x T x D
            pred: Dict containing predictions from multiple teachers
            target: Dict containing targets corresponding to each teacher
            return_other: Flag to indicate if additional losses should be returned
        """
        # Initialize variables to accumulate overall losses
        total_loss = 0
        total_rec_loss = 0
        total_sim_loss = 0
        total_feat_pen = 0
        teacher_total_losses = []  # List to store individual teacher's total loss for PCGrad

        rec_layer_loss_dict = {}
        sim_layer_loss_dict = {}

        # Iterate over each teacher's predictions and targets
        for teacher_key in target.keys():
            teacher_pred = pred[teacher_key]
            teacher_target = target[teacher_key]
            teacher_target = teacher_target[:, :, :teacher_pred.shape[2], :]
            # Align predictions if using ssast_frame
            if 'ssast_frame' in self.teacher_models:
                aligned_preds = []
                aligned_targets = []
                for i in range(teacher_pred.shape[1]):
                    align_teacher, align_student = self.temporal_alignment(teacher_target[:, i, :, :], teacher_pred[:, i, :, :])
                    aligned_preds.append(align_student.unsqueeze(1))
                    aligned_targets.append(align_teacher.unsqueeze(1))
                teacher_pred = torch.cat(aligned_preds, dim=1)
                teacher_target = torch.cat(aligned_targets, dim=1)

            assert teacher_pred.shape == teacher_target.shape, (teacher_pred.shape, teacher_target.shape)
            # print(teacher_pred.shape, teacher_target.shape, teacher_key)
            # Calculate reconstruction loss
            if self.train_on_all_data:
                rec_loss = self.loss_func(teacher_pred, teacher_target)  # B x N x T x D
                if teacher_key == 'ssast_frame':
                    weighted_rec_loss = rec_loss.mean() * 0.05
                else:
                    weighted_rec_loss = rec_loss.mean()
            else:
                teacher_pred_for_spec_domain = []
                teacher_target_for_spec_domain = []
                for i, domain in enumerate(sample_domain):
                    if domain == teacher_key or (domain=='hubert_base' and (teacher_key=='whisper_small' or teacher_key=='wavlm')) or teacher_key=='emotion2vec' or teacher_key=='hubert_base':
                        teacher_pred_for_spec_domain.append(teacher_pred[i])
                        teacher_target_for_spec_domain.append(teacher_target[i])
                # print(sample_domain, len(teacher_pred_for_spec_domain))
                if len(teacher_pred_for_spec_domain) > 0:
                    # print(teacher_key, len(teacher_pred_for_spec_domain))
                    teacher_pred = torch.stack(teacher_pred_for_spec_domain)
                    teacher_target = torch.stack(teacher_target_for_spec_domain)
                    rec_loss = self.loss_func(teacher_pred, teacher_target)
                    if teacher_key == 'ssast_frame':
                        weighted_rec_loss = rec_loss.mean() * 0.05
                    elif teacher_key == 'mert_v0_public':
                        weighted_rec_loss = rec_loss.mean() * self.config.mert_weight
                    elif teacher_key == 'hubert_base':
                        weighted_rec_loss = rec_loss.mean() * self.config.hubert_weight
                    elif teacher_key == 'wavlm':
                        weighted_rec_loss = rec_loss.mean() * self.config.wavlm_weight 
                    elif teacher_key == 'emotion2vec':
                        weighted_rec_loss = rec_loss.mean() * self.config.emotion2vec_weight
                    else:
                        weighted_rec_loss = rec_loss.mean()
                else:
                    print(f"No matching predictions found for teacher: {teacher_key}")
                    weighted_rec_loss = 0
                    rec_loss = torch.zeros_like(teacher_target)
                    
            total_rec_loss += weighted_rec_loss

            # Calculate similarity loss if applicable
            if self.cosine_loss > 0:
                sim_loss = -F.logsigmoid(F.cosine_similarity(teacher_pred, teacher_target, dim=-1))
                if teacher_key == 'mert_v0_public':    
                    #print('multiplying weight', self.config.mert_weight)
                    total_sim_loss += sim_loss.mean() * self.config.mert_weight
                elif teacher_key == 'hubert_base':
                    total_sim_loss += sim_loss.mean() * self.config.hubert_weight               
                elif teacher_key == 'wavlm':
                    total_sim_loss += sim_loss.mean() * self.config.wavlm_weight
                elif teacher_key == 'emotion2vec':
                    total_sim_loss += sim_loss.mean() * self.config.emotion2vec_weight
                else:
                    total_sim_loss += sim_loss.mean()
            else:
                sim_loss = 0

            # Calculate feature penalty loss
            feat_pen = feat.float().pow(2).mean()
            total_feat_pen += feat_pen

            # Calculate the total loss for this teacher and add it to the list
            teacher_total_loss = (
                weighted_rec_loss
                + feat_pen * self.config.feat_pen_loss
                + sim_loss.mean() * self.cosine_loss
            )
            teacher_total_losses.append(teacher_total_loss)

            # Store layer-wise loss if return_other is True
            if return_other:
                with torch.no_grad():
                    rec_layer_loss = rec_loss.mean((0, 2, 3)) if rec_loss.numel() > 0 else 0
                    if teacher_key == 'ssast_frame':
                        rec_layer_loss *= 0.05
                rec_layer_loss_dict[teacher_key] = rec_layer_loss

                if self.cosine_loss > 0:
                    with torch.no_grad():
                        sim_layer_loss = sim_loss.mean((0, 2))
                    sim_layer_loss_dict[teacher_key] = sim_layer_loss
                else:
                    sim_layer_loss_dict[teacher_key] = None
            else:
                rec_layer_loss_dict[teacher_key] = None
                sim_layer_loss_dict[teacher_key] = None

        # Sum up the total loss components for overall training
        total_loss = (
            total_rec_loss
            + total_feat_pen * self.config.feat_pen_loss
            + total_sim_loss * self.cosine_loss
        )

        # Return individual teacher total losses for PCGrad, as well as other components
        return total_loss, teacher_total_losses, total_rec_loss, rec_layer_loss_dict, total_feat_pen, total_sim_loss, sim_layer_loss_dict
