"""
    Upstream expert for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import io
import yaml

from ..interfaces import UpstreamBase
from torch.nn.utils.rnn import pad_sequence
# from .builder import PretrainedDistiller
from .model import MultiDistillerModel, MultiDistillerConfig
from transformers import AutoModel, AutoConfig
import torch

class UpstreamExpert(UpstreamBase):
    """
    The Distiller wrapper
    """

    def __init__(self, ckpt, model_config=None, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # with open("./upstream/multi_distiller/config_model.yaml", "r") as f:
        self.topline = False
        '''
        if self.topline:
            self.hubert = torch.hub.load("s3prl/s3prl", 'hubert_base' ).to(self.device)
            temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
            temp_config.output_hidden_states = True
            self.mert = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).to(self.device)
        '''

        if not self.topline:
            self.model = MultiDistillerModel(MultiDistillerConfig(model_config["multi_distiller"]))
            self.model.load_state_dict(torch.load(ckpt,weights_only=False)["Distiller"])
        
        if self.topline:
            with open('/work/u8786328/john/temp/result/pretrain/distill_wavlm_init_on_wavlm/config_model.yaml', 'r') as file:
                distillwavlm_config = yaml.load(file, Loader=yaml.FullLoader)

            self.distillwavlm = MultiDistillerModel(MultiDistillerConfig(distillwavlm_config["multi_distiller"]))
            self.distillwavlm.load_state_dict(torch.load('/work/u8786328/john/temp/result/pretrain/distill_wavlm_init_on_wavlm/states-epoch-17.ckpt')["Distiller"])

            with open('/work/u8786328/john/temp/result/pretrain/distill_hubert_2_layer_librispeech960/config_model.yaml', 'r') as file:
                distillhubert_config = yaml.load(file, Loader=yaml.FullLoader)

            self.distillhubert = MultiDistillerModel(MultiDistillerConfig(distillhubert_config["multi_distiller"]))
            self.distillhubert.load_state_dict(torch.load('/work/u8786328/john/temp/result/pretrain/distill_hubert_2_layer_librispeech960/states-epoch-17.ckpt')["Distiller"])
            #print('distillhubert: ', self.distillhubert)
            with open('/work/u8786328/john/temp/result/pretrain/distill_mert_2_layer_music4all/config_model.yaml', 'r') as file:
                distillmert_config = yaml.load(file, Loader=yaml.FullLoader)
            # print(distillmert_config)
            self.distillmert = MultiDistillerModel(MultiDistillerConfig(distillmert_config["multi_distiller"]))
            self.distillmert.load_state_dict(torch.load('/work/u8786328/john/temp/result/pretrain/distill_mert_2_layer_music4all/states-epoch-43.ckpt')["Distiller"])
            #print('distillmert:', self.distillmert)

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs, no_pred=False):
        x_pad_batch = pad_sequence(wavs, batch_first=True)

        # Create the padding mask for 16 kHz
        x_lens = torch.LongTensor([wave.shape[-1] for wave in wavs])  # Sequence lengths for 16 kHz
        pad_mask = torch.ones(x_pad_batch.shape[:2], dtype=torch.bool).to(self.device)  # (batch_size, seq_len)
        for idx in range(x_pad_batch.shape[0]):
            pad_mask[idx, x_lens[idx]:] = 0  # Mask out padding regions with zeros

        if self.topline:
            '''
            hubert_hiddens = self.hubert(x_pad_batch)["hidden_states"]
            #hubert_hiddens = torch.stack(hubert_hiddens, dim=0)
            mert_hiddens = self.mert(x_pad_batch)["hidden_states"]
            #mert_hiddens = torch.stack(mert_hiddens, dim=0)
            # topline_feats = []
            # topline_feats = torch.cat(hubert_hiddens[1:] + mert_hiddens[1:], dim=0)         
            selected_hubert_hiddens = [hubert_hiddens[i] for i in [4, 8, 12]]
            selected_mert_hiddens = [mert_hiddens[i] for i in [4, 8, 12]]
            selected_hubert_hiddens_tensor = torch.stack(selected_hubert_hiddens)
            selected_mert_hiddens_tensor = torch.stack(selected_mert_hiddens)
            #for i, hidden in enumerate(selected_hubert_hiddens):
                #print(f"Shape of selected_hubert_hiddens[{i}]: {hidden.shape}")
            # Concatenate the selected layers
            #print(selected_hubert_hiddens_tensor.shape)
            #topline_feats = torch.cat([selected_hubert_hiddens_tensor + selected_mert_hiddens_tensor], dim=0)
            topline_feats = selected_hubert_hiddens + selected_mert_hiddens
            # print(len(topline_feats))
            states = {"topline": topline_feats,}
            return states
            '''
            pad_mask_token = pad_mask
            #_, wavlm_feat_final, pred, pad_mask, wavlm_layer_hidden = self.distillwavlm(x_pad_batch, pad_mask=pad_mask_token, get_hidden=True, no_pred=no_pred)
            _, hubert_feat_final, pred, pad_mask, hubert_layer_hidden = self.distillhubert(x_pad_batch, pad_mask=pad_mask_token, get_hidden=True, no_pred=no_pred)
            _, mert_feat_final, pred, pad_mask, mert_layer_hidden = self.distillmert(x_pad_batch, pad_mask=pad_mask_token, get_hidden=True, no_pred=no_pred)
            #topline_feats = wavlm_layer_hidden + mert_layer_hidden
            hidden_concat = [
                torch.cat([hubert_layer_hidden[i], mert_layer_hidden[i]], dim=-1)
                for i in range(len(hubert_layer_hidden))
            ]
            #print(len(topline_feats))
            #print(topline_feats)
            states = {"topline": hidden_concat,}
            return states

        _, feat_final, pred, pad_mask, layer_hidden = self.model(
            x_pad_batch, pad_mask=pad_mask, get_hidden=True, no_pred=no_pred
        )
            
        # pred: B x N x T x D
        # if not no_pred:
        #     hidden_feats = pred.transpose(0, 1).split(1, 0)
        #     hidden_feats = [hid.squeeze(0) for hid in hidden_feats]
        # else:
        #     
        hidden_feats = []
        #print(len(feat_final), len(layer_hidden))
        hidden_feats = [feat_final] + layer_hidden + hidden_feats
        #print('hidden feats', feat_final.shape)
        states = {
            "last_hidden_state": None if no_pred else hidden_feats[-1],
            "hidden_states": hidden_feats,
            "pad_mask": pad_mask,
            "paper": layer_hidden[-1],  # DistilHuBERT: https://arxiv.org/abs/2110.01900
        }

        return states

