import torch
import torchaudio
import torchaudio.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

waveform, sample_rate = torchaudio.load("music/yesand.wav")
waveform = waveform.to(device)
num_channels, num_frames = waveform.shape

resample_rate = sample_rate/4

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=64)

print(f"{resample_rate} Hz <- {sample_rate} Hz")

torch.save(resampled_waveform, "./downsampled.pt")
torchaudio.save(uri="./downsampled.wav", src=resampled_waveform.detach().cpu(), sample_rate=int(resample_rate), format="wav")