import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


seq_len = 500

class InterpolationModel(nn.Module):
    def __init__(self, hidden_size=256):
        super(InterpolationModel, self).__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)

        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.FloatTensor, h_0=None, c_0=None):
        x = x.unsqueeze(-1)
        if h_0 is None or c_0 is None:
            lstm_out, (h_n, c_n) = self.lstm(x)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        lstm_out = lstm_out[:, -1, :]
        x = self.net(lstm_out)
        return x, (h_n, c_n)
    
class MusicDataset(Dataset):
    def __init__(self, root, seq_len: int, device="cpu"):
        self.root = root
        self.files = os.listdir(root)
        self.files = [os.path.join(root, d) for d in self.files]
        self.songs = []
        self.songidx = []
        self.total_waves = 0
        self.device = device
        self.seq_len = seq_len

        for d in self.files:
            waveform, sample_rate = torchaudio.load(d)
            num_channels, num_frames = waveform.shape
            for channel in range(num_channels):
                self.songidx.append(self.total_waves)
                self.songs.append(waveform[channel])
                self.total_waves += num_frames
        
        logging.info(f"Total number of waves: {self.total_waves}")
        logging.info(f"Total number of songs: {len(self.songs)}")
        logging.info(f"Data Type: {tuple((self.songs[i].dtype for i in range(len(self.songs))))}")
    
    def __len__(self):
        return self.total_waves
    
    def __getitem__(self, idx):
        # Find song with the index lower but closest to idx
        song = None
        for i, startidx in enumerate(self.songidx):
            if startidx <= idx:
                song = (i, startidx)
            else:
                break
        
        waveform = self.songs[song[0]].to(self.device)

        # Get next 20 frames
        seq = waveform[idx - song[1] : idx - song[1] + self.seq_len * 2]
        # Add padding if necessary
        if len(seq) < self.seq_len * 2:
            padding = torch.zeros(self.seq_len * 2 - len(seq), device=self.device)
            seq = torch.cat((seq, padding))

        X = seq[::2]
        y = torch.tensor((seq[-3],), device=self.device)

        return X, y
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    bs = 128 if device.type == "cuda" else 8
    n_epochs = 1
    max_batches = None

    musicDataset = MusicDataset("music", device="cpu", seq_len=seq_len)
    dset_train, dset_val = random_split(musicDataset, [0.8, 0.2])

    model = InterpolationModel()
    model.to(device)
    model = torch.compile(model)

    train = DataLoader(dset_train, batch_size=bs, shuffle=True, num_workers=0 if device.type == "cuda" else 0)
    val = DataLoader(dset_val, batch_size=bs, shuffle=False, num_workers=4 if device.type == "cuda" else 0)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001, eps=1e-10)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)
    criterion = nn.MSELoss()

    print("Training...")
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        val_losses = []
        for i, (X, y) in enumerate(tqdm(train, mininterval=1, total=max_batches)):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)[0]
            l = criterion(pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_losses.append(l.item())

            print(f"\033[F\033[KLoss: {l.item()}", flush=False)

            if max_batches and i + 1 == max_batches:
                break

        print(f"\nEpoch: {epoch}, Loss: {round(sum(train_losses) / len(train_losses), 6)}")
        

        scheduler.step()
        
        with torch.no_grad():
            model.eval()
            for i, (X, y) in enumerate(tqdm(val, mininterval=1, total=max_batches//2 if max_batches else None)):
                X = X.to(device)
                y = y.to(device)

                pred = model(X)[0]
                l = criterion(pred, y)

                val_losses.append(l.item())

                if max_batches and i + 1 == max_batches // 2:
                    break

            print(f"Epoch: {epoch}, Val Loss: {round(sum(val_losses) / len(val_losses), 6)}\n")
        
        # Print graph
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.legend()
        plt.show()

    model.eval()

    torch.save(model, "model.pt")
