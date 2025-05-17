import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MaestroDataset
from model import CRNNTranscriber
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train():
    print(f"Using device: {device}")
    writer = SummaryWriter(log_dir="runs/experiment1")
    dataset = MaestroDataset(
        csv_path="../dataset/maestro-v3.0.0/maestro-v3.0.0/maestro-v3.0.0.csv",
        audio_dir="../dataset/maestro-v3.0.0/maestro-v3.0.0"
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model = CRNNTranscriber().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        for i, (mel, target) in enumerate(loop):
            mel, target = mel.to(device), target.transpose(1, 2).to(device)
            optimizer.zero_grad()
            output = model(mel)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            writer.add_scalar("Loss/train", loss.item(), epoch * len(loader) + i)

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # 학습 끝나고 모델 저장
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/best_model.pt")
    print("Model saved to checkpoints/best_model.pt")
    writer.close()


if __name__ == "__main__":
    train()