import torch
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
from jiwer import cer, wer
import random
import numpy as np
from text_utils import CHARS, int_to_text
from model import load_model
from data_loader import LibriSpeechDataset, collate_audio_fn
from torch.utils.data import DataLoader
from configs import *
from torch import nn
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def run_test_inference(model, loader, device, num_samples=5):
    """Esegue l'inferenza su alcuni campioni casuali del test set per vedere i risultati"""
    model.eval()

    
    scratch_path = "/scratch.hpc/marcello.spagnoli2/models/lm"
    os.makedirs(scratch_path, exist_ok=True)

    # Imposta la variabile d'ambiente per dire a Torch dove scaricare
    os.environ["TORCH_HOME"] = scratch_path
    files = download_pretrained_files("librispeech-3-gram")
    print("Calcolo loss, CER e WER su tutto il test set...")

    # necessario per il decoder CTC di torchaudio, che si aspetta i token in minuscolo
    decoder_tokens = [c.lower() for c in CHARS]
    if " " in decoder_tokens:
        idx_spazio = decoder_tokens.index(" ")
        decoder_tokens[idx_spazio] = "|"

    print(f"Decoder tokens: {decoder_tokens}")
    # Crea il decoder CTC con beam search
    decoder = ctc_decoder(
        lexicon=files.lexicon,      # Il lessico è fondamentale con un LM
        tokens=decoder_tokens,      # La tua lista di caratteri
        lm=files.lm,                # Percorso al file .arpa o .bin del modello
        blank_token="-",
        sil_token="|",
        nbest=1,
        beam_size=30,               # Con un LM, un beam_size maggiore (20-50) aiuta
        lm_weight=0.3,              # Peso dell'influenza del linguaggio (da tunare)
        word_score=10,            # Penalità/premio per ogni parola trovata
    )

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Raccogli campioni e calcola loss in una sola passata
    all_samples = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            mels, input_lengths, targets, target_lengths = [b.to(device) for b in batch]
            logits = model(mels)
            log_probs = torch.log_softmax(logits, dim=-1).cpu()
            
            # Calcola loss per questo batch
            logits_for_loss = logits.permute(1, 0, 2)
            loss = criterion(logits_for_loss, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            num_batches += 1
            
            for j in range(mels.shape[0]):
                sample_log_probs = log_probs[j:j+1, :input_lengths[j], :] # [1, T, C] -> [T, C]
                results = decoder(sample_log_probs)
                
                if len(results[0]) > 0:
                    pred_tokens = results[0][0].tokens.tolist()
                    pred_text = int_to_text(pred_tokens).upper()
                else:
                    pred_text = ""
                
                # Estrae il testo reale dai target appiattiti
                start_idx = sum(target_lengths[:j]).item() if j > 0 else 0
                end_idx = start_idx + target_lengths[j].item()
                actual_text = int_to_text(targets[start_idx:end_idx].tolist())
                
                all_samples.append({
                    'actual': actual_text,
                    'predicted': pred_text,
                    'cer': cer(actual_text, pred_text),
                    'wer': wer(actual_text, pred_text)
                })
    
    avg_test_loss = total_loss / num_batches
    cer_avg = np.mean([s['cer'] for s in all_samples])
    wer_avg = np.mean([s['wer'] for s in all_samples])
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Average CER: {cer_avg:.4f}")
    print(f"Average WER: {wer_avg:.4f}")

    
    print("\n--- ESEMPI DI PREDIZIONE SUL TEST SET (RANDOM) ---")
    # Seleziona n campioni casuali
    num_to_show = min(num_samples, len(all_samples))
    random_indices = random.sample(range(len(all_samples)), num_to_show)
    
    for idx, sample_idx in enumerate(sorted(random_indices), 1):
        sample = all_samples[sample_idx]
        print(f"Campione {idx} (indice {sample_idx}):")
        print(f"  Vero: {sample['actual']}")
        print(f"  Pred: {sample['predicted']}")
        print(f"  CER: {sample['cer']:.4f}")
        print(f"  WER: {sample['wer']:.4f}")
        print("-" * 30)


if __name__ == "__main__":
    set_seed(42)
    
    test_ds = LibriSpeechDataset(DATA_DIR, "test-clean", download=DOWNLOAD_DATASET)

    test_loader = DataLoader(
            test_ds, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            collate_fn=collate_audio_fn
        )

    model = load_model(MODELS_DIR / "best_model.pth", DEVICE)

    run_test_inference(model, test_loader, DEVICE, num_samples=5)