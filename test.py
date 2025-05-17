import torch
import librosa
import numpy as np
from model import CRNNTranscriber
from utils import piano_roll_to_midi
from music21 import converter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def wav_to_score(wav_path):
    # 1. wav → mel 변환
    y, sr = librosa.load(wav_path, sr=16000)
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=229)
    mel_db = librosa.power_to_db(mel)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)  # 0~1 정규화
    mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).to(device)  # [1, n_mels, T]

    # 2. 모델 추론
    model = CRNNTranscriber().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        output = model(mel_tensor)  # [1, T, 128]
        piano_roll = output[0].cpu().numpy()  # [T, 128]

    # 3. 피아노롤 → MIDI
    midi = piano_roll_to_midi(piano_roll)
    midi_path = "result.mid"
    midi.write(midi_path)

    # 4. MIDI → 악보 띄우기
    score = converter.parse(midi_path)
    score.write('musicxml.png', 'result.png')  # 악보를 PNG로 저장
    print("result.png로 악보 이미지가 저장되었습니다.")

if __name__ == "__main__":
    wav_to_score("test.wav")  # test.wav 파일을 같은 폴더에 두세요