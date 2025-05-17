import torch
import torch.nn as nn

class CRNNTranscriber(nn.Module):
    def __init__(self, n_mels=229, hidden_size=256, num_notes=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),  # mel만 2배 줄임
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1))  # mel만 2배 줄임
        )
        self.rnn = nn.LSTM(64 * (n_mels // 4), hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_notes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, Mels, Time]
        x = self.cnn(x)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # [B, T, C*F]
        x, _ = self.rnn(x)
        x = self.fc(x)
        return self.sigmoid(x)  # output: [B, T, 128]
