import torch
from torch.utils.data import DataLoader
from dataset import MaestroDataset
from model import CRNNTranscriber
from utils import piano_roll_to_midi
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(model_path, output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MaestroDataset(
        csv_path="../dataset/maestro-v3.0.0/maestro-v3.0.0/maestro-v3.0.0.csv",
        audio_dir="../dataset/maestro-v3.0.0/maestro-v3.0.0"
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = CRNNTranscriber().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    precision_list, recall_list, f1_list = [], [], []

    with torch.no_grad():
        for i, (mel, target) in enumerate(loader):
            if i >= num_samples:
                break

            mel = mel.to(device)
            target = target.transpose(1, 2).squeeze(0).cpu().numpy()  # [T, 128]
            output = model(mel).squeeze(0).cpu().numpy()              # [T, 128]

            # Save prediction as MIDI
            midi = piano_roll_to_midi(output)
            midi.write(os.path.join(output_dir, f"pred_{i}.mid"))

            # Evaluation
            pred_bin = (output > 0.5).astype(np.uint8)
            tgt_bin = (target > 0.5).astype(np.uint8)
            if pred_bin.shape != tgt_bin.shape:
                min_len = min(pred_bin.shape[0], tgt_bin.shape[0])
                pred_bin = pred_bin[:min_len]
                tgt_bin = tgt_bin[:min_len]

            precision = precision_score(tgt_bin.flatten(), pred_bin.flatten(), zero_division=0)
            recall = recall_score(tgt_bin.flatten(), pred_bin.flatten(), zero_division=0)
            f1 = f1_score(tgt_bin.flatten(), pred_bin.flatten(), zero_division=0)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

            print(f"[{i}] Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    print("\n=== Average Metrics ===")
    print(f"Precision: {np.mean(precision_list):.4f}")
    print(f"Recall:    {np.mean(recall_list):.4f}")
    print(f"F1 Score:  {np.mean(f1_list):.4f}")

if __name__ == "__main__":
    evaluate(
        model_path="checkpoints/best_model.pt",
        output_dir="outputs/midis",
        num_samples=10
    )
