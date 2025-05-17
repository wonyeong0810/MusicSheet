import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pretty_midi
import os
import pandas as pd

class MaestroDataset(Dataset):
    def __init__(self, csv_path, audio_dir, sr=16000, hop_length=512, n_mels=229, duration=10):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.sr = sr
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.duration = duration

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['audio_filename'])
        midi_path = os.path.join(self.audio_dir, row['midi_filename'])

        # Load audio
        y, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, hop_length=self.hop_length, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel).astype(np.float32)

        # Load MIDI -> Piano roll
        midi = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = midi.get_piano_roll(fs=self.sr // self.hop_length).astype(np.float32)
        piano_roll = (piano_roll > 0).astype(np.float32)

        # Cut or pad to same size
        if piano_roll.shape[1] > mel_db.shape[1]:
            piano_roll = piano_roll[:, :mel_db.shape[1]]
        else:
            pad_width = mel_db.shape[1] - piano_roll.shape[1]
            piano_roll = np.pad(piano_roll, ((0,0), (0, pad_width)), 'constant')

        return torch.tensor(mel_db), torch.tensor(piano_roll)
