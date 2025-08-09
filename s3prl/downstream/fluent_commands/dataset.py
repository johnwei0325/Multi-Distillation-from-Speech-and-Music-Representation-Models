import os
import random
from pathlib import Path

import torch
import torchaudio
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

SAMPLE_RATE = 16000
EXAMPLE_WAV_MAX_SEC = 10


class FluentCommandsDataset(Dataset):
    def __init__(self, df, base_path, Sy_intent, **kwargs):
        self.df = df
        self.base_path = base_path
        self.max_length = SAMPLE_RATE * EXAMPLE_WAV_MAX_SEC
        self.Sy_intent = Sy_intent
        self.upstream_name = kwargs['upstream']
        self.features_path = kwargs['features_path']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.base_path, self.df.loc[idx].path)
        wav, sr = torchaudio.load(wav_path)

        wav = wav.squeeze(0)

        label = []

        for slot in ["action", "object", "location"]:
            value = self.df.loc[idx][slot]
            label.append(self.Sy_intent[slot][value])
        
        if self.features_path:
            feature_path = os.path.join(self.features_path, self.upstream_name, f"{Path(wav_path).stem}.pt")
            if os.path.exists(feature_path):
                feature = torch.load(feature_path)
                return feature, label, True

        return wav.numpy(), np.array(label), Path(wav_path).stem

    def collate_fn(self, samples):
        return zip(*samples)


class FluentCommandsWithSingerDataset(FluentCommandsDataset):
    """Fluent commands dataset with singer audio mixing capability."""

    def __init__(self, df, base_path, Sy_intent, **kwargs):
        super().__init__(df, base_path, Sy_intent, **kwargs)
        
        # Initialize singer dataset with default paths
        from ..vocalset_singer_id.dataset import SingerDataset
        self.singer_dataset = SingerDataset(
            "/home/johnwei743251/mdd/dataset/VocalSet",  # Default singer file path
            "/home/johnwei743251/mdd/dataset/singer_id/meta_data/data",  # Default singer meta data path
            "test",
            upstream=kwargs.get("upstream"),
            features_path=kwargs.get("features_path"),
        )
        
        # Audio processing parameters
        self.sample_rate = SAMPLE_RATE
        self.sample_duration = EXAMPLE_WAV_MAX_SEC
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
        # Get the base fluent commands data
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

