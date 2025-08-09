from random import randint
from pathlib import Path
import os
import torch
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
import torchaudio
import numpy as np
import torch.nn.functional as F

CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "_unknown_",
    "_silence_",
]

EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]


class SpeechCommandsBaseDataset(Dataset):
    """12-class Speech Commands base dataset."""

    def __init__(self):
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}
        self.class_num = 12
        self.data = []

    def __getitem__(self, idx):
        class_name, audio_path = self.data[idx]
        wav, _ = apply_effects_file(str(audio_path), EFFECTS)
        wav = wav.squeeze(0).numpy()
        fileid = "-".join(Path(audio_path).parts[-2:])
        return wav, self.class2index[class_name], fileid

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        return zip(*samples)


class SpeechCommandsDataset(SpeechCommandsBaseDataset):
    """Training and validation dataset."""

    def __init__(self, data_list, **kwargs):
        super().__init__()

        data = [
            (class_name, audio_path)
            if class_name in self.class2index.keys()
            else ("_unknown_", audio_path)
            for class_name, audio_path in data_list
        ]
        data += [
            ("_silence_", audio_path)
            for audio_path in Path(
                kwargs["speech_commands_root"], "_background_noise_"
            ).glob("*.wav")
        ]

        class_counts = {class_name: 0 for class_name in CLASSES}
        for class_name, _ in data:
            class_counts[class_name] += 1

        sample_weights = [
            len(data) / class_counts[class_name] for class_name, _ in data
        ]

        self.data = data
        self.sample_weights = sample_weights
        self.upstream_name = kwargs['upstream']
        self.features_path = kwargs['features_path']

    def __getitem__(self, idx):
        wav, label, stem = super().__getitem__(idx)

        # _silence_ audios are longer than 1 sec.
        if label == self.class2index["_silence_"]:
            random_offset = randint(0, len(wav) - 16000)
            wav = wav[random_offset : random_offset + 16000]
        
        if self.features_path:
            feature_path = os.path.join(self.features_path, self.upstream_name, f"{stem}.pt")
            if os.path.exists(feature_path):
                feature = torch.load(feature_path)
                return feature, label, True

        return wav, label, stem


class SpeechCommandsTestingDataset(SpeechCommandsBaseDataset):
    """Testing dataset."""

    def __init__(self, **kwargs):
        super().__init__()

        self.data = [
            (class_dir.name, audio_path)
            for class_dir in Path(kwargs["speech_commands_test_root"]).iterdir()
            if class_dir.is_dir()
            for audio_path in class_dir.glob("*.wav")
        ]


class SpeechCommandsTestingWithSingerDataset(SpeechCommandsTestingDataset):
    """Testing dataset with singer audio mixing capability."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize singer dataset with default paths
        from ..vocalset_singer_id.dataset import SingerDataset
        from ..vocalset_technique_id.dataset import VocalTechniqueDataset
        self.singer_dataset = VocalTechniqueDataset(
            "/home/johnwei743251/mdd/dataset/VocalSet",  # Default singer file path
            "/home/johnwei743251/mdd/dataset/vocal_id/meta_data/data",  # Default singer meta data path
            "test",
            upstream=kwargs.get("upstream"),
            features_path=kwargs.get("features_path"),
        )
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.sample_duration = 1  # 1 second of audio to match speech commands
        self.sample_length = self.sample_rate * self.sample_duration

    def load_and_process_audio(self, audio_path, is_singer=False):
        """Load and process audio file with resampling and duration adjustment"""
        # Convert PosixPath to string if necessary
        audio_path = str(audio_path)
        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        audio = wav.squeeze()

        # Handle audio duration
        if audio.shape[0] <= self.sample_length:
            # Pad if shorter
            audio = F.pad(audio, (0, self.sample_length - audio.shape[0]), 'constant', 0)
        else:
            # Random crop if longer
            random_start = np.random.randint(0, audio.shape[0] - self.sample_length)
            audio = audio[random_start:random_start + self.sample_length]

        return audio

    def mix_audio(self, singer_audio, speech_audio):
        """Mix two audio signals with emphasis on speech data"""
        # Ensure both audios are the same length
        if singer_audio.shape[0] != speech_audio.shape[0]:
            min_length = min(singer_audio.shape[0], speech_audio.shape[0])
            singer_audio = singer_audio[:min_length]
            speech_audio = speech_audio[:min_length]
        
        # Normalize both signals
        singer_audio = singer_audio / torch.max(torch.abs(singer_audio))
        speech_audio = speech_audio / torch.max(torch.abs(speech_audio))
        
        # Calculate RMS energy
        singer_rms = torch.sqrt(torch.mean(singer_audio ** 2))
        speech_rms = torch.sqrt(torch.mean(speech_audio ** 2))
        
        # Target speech-to-singing ratio (in dB) - increased to emphasize speech more
        target_ratio_db = np.random.uniform(6, 9)  # Speech 6-9dB louder than singing
        target_ratio = 10 ** (target_ratio_db / 20)
        
        # Calculate scaling factors to achieve target ratio
        current_ratio = speech_rms / (singer_rms + 1e-6)
        if current_ratio < target_ratio:
            # Need to increase speech energy
            speech_scale = target_ratio / current_ratio
            singer_scale = 0.5  # Reduce singer audio energy
        else:
            # Need to decrease singing energy
            singer_scale = 0.5  # Keep singer audio low
            speech_scale = 1.0
        
        # Apply scaling with some random variation
        speech_scale *= np.random.uniform(0.95, 1.05)  # Less variation for speech
        singer_scale *= np.random.uniform(0.3, 0.5)    # More variation for singing
        
        # Mix the signals
        mixed_audio = singer_audio * singer_scale + speech_audio * speech_scale
        
        # Normalize the mixed signal to prevent clipping
        max_val = torch.max(torch.abs(mixed_audio))
        if max_val > 0.95:  # If close to clipping
            mixed_audio = mixed_audio * (0.95 / max_val)
        
        return mixed_audio

    def __getitem__(self, idx):
        # Get the base speech command data using parent's method
        wav, label, stem = super().__getitem__(idx)
        
        # If features are available, return them directly
        if isinstance(wav, torch.Tensor) and wav.dim() > 1:  # This indicates it's a feature tensor
            return wav, label, stem
            
        # Convert to tensor if it's numpy array
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        
        # Get random singer audio
        singer_idx = np.random.randint(0, len(self.singer_dataset))
        singer_data = self.singer_dataset[singer_idx]
        singer_audio_path = os.path.join(self.singer_dataset.audio_dir, "audio", singer_data[2])
        
        # Load and process singer audio
        singer_audio = self.load_and_process_audio(singer_audio_path, is_singer=True)
        
        # Mix the audios
        mixed_audio = self.mix_audio(singer_audio, wav)
        #mixed_audio = wav
        return mixed_audio.numpy(), label, stem

