import torch
import torchaudio
from torch.utils.data import Dataset
from text_utils import text_to_int

class LibriSpeechDataset(Dataset):
    def __init__(self, root, url, download=False):
        """
        Args:
            root: Directory radice dove cercare/scaricare il dataset
            url: Nome del subset (es. "dev-clean", "test-clean", "train-clean-100")
            download: Se True, scarica il dataset se non presente (default: False per cluster)
        """
        print(f"Caricamento dataset: {url} da {root}")
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=download)
        dataset_len = len(self.dataset)
        print(f"Dataset caricato: {dataset_len} campioni trovati")
        
        if dataset_len == 0:
            raise ValueError(
                f"Dataset vuoto! Verifica che {url} sia presente in {root}/LibriSpeech/\n"
                f"Controlla con: ls -la {root}/LibriSpeech/{url}/"
            )
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, _, transcript, *_ = self.dataset[idx]
        waveform = waveform / (waveform.abs().max() + 1e-9)
        mel = self.amplitude_to_db(self.mel_transform(waveform))
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        mel = mel.squeeze(0).permute(1, 0) # [T, n_mels]
        return mel, text_to_int(transcript)

def collate_audio_fn(batch):
    mels, targets_list, input_lengths, target_lengths = [], [], [], []
    for mel, encoded in batch:
        mels.append(mel)
        targets_list.extend(encoded)
        input_lengths.append(mel.shape[0])
        target_lengths.append(len(encoded))

    mels_padded = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=0.0)
    return mels_padded, torch.tensor(input_lengths), torch.tensor(targets_list), torch.tensor(target_lengths)