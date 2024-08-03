import torch
import torchaudio
from torchaudio.io import StreamReader
from torchaudio.io import StreamWriter
from train import InterpolationModel
import tqdm


bs = 16
seq_len = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = torch.load("model.pt", map_location=device, weights_only=False)
model.eval()


waveform, sample_rate = torchaudio.load("out.wav")
num_channels, num_frames = waveform.shape

#new_waveform = [[], []]
#new_waveform[0] = tuple(waveform[0][::4])
#new_waveform[1] = tuple(waveform[1][::4])
#waveform = torch.tensor(new_waveform)
#del new_waveform
#num_frames = num_frames // 4

waveform = waveform.to(device)

#ws = StreamWriter("out.wav")
#ws.add_audio_stream(sample_rate*2, num_channels)
out = [[] for _ in range(num_channels)]

print("Interpolating...")


with torch.no_grad():
    frame_count = 0
    
    processQueue = []
    for i in tqdm.tqdm(range(len(waveform[0]))):
        if i < seq_len:
            padding = torch.zeros(seq_len - i - 1, device=device)
            seq = []
            for channel in range(num_channels):
                tensor = torch.cat((padding, waveform[channel][0 : i + 1]))
                seq.append((tensor, channel))
        else:
            # Get previous seq_len frames
            seq = [((waveform[channel][i - seq_len + 1 : i + 1]), channel) for channel in range(num_channels)]

        processQueue.extend(seq)

        if len(processQueue) == bs or len(waveform[0]) - 1 == i:
            preds = model(torch.stack(list(x[0] for x in processQueue))).cpu()

            for i, sequence in enumerate(processQueue):
                #chunk.append((sequence[0][-1], sequence[1]))
                #chunk.append((preds[i], sequence[1]))
                out[sequence[1]].append(sequence[0][-2])
                out[sequence[1]].append(preds[i])

            #frame_count += len(chunk[0]) // num_channels
            #f.write_audio_chunk(0, torch.tensor(chunk))
            #out.extend(chunk)
            processQueue = []

    frame_count = len(out[0])
    print(f"Number of frames in input: {num_frames}")
    print(f"Number of frames in output: {frame_count}")

    # Calculate new sample rate ratio
    ratio = 2
    upsampled_sample_rate = sample_rate * ratio
    print(f"{sample_rate} Hz -> {upsampled_sample_rate} Hz  ({ratio}x)")

    torchaudio.save('out.wav', torch.tensor(out), sample_rate=int(upsampled_sample_rate))