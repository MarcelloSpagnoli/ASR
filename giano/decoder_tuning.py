import os

import numpy as np
import optuna
import torch
from jiwer import wer
from torch.utils.data import DataLoader
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files

from configs import *
from data_loader import LibriSpeechDataset, collate_audio_fn
from model import load_model
from text_utils import CHARS, int_to_text


def build_decoder(beam_size, lm_weight, word_score, files, decoder_tokens):
    return ctc_decoder(
        lexicon=files.lexicon,
        tokens=decoder_tokens,
        lm=files.lm,
        blank_token="-",
        sil_token="|",
        nbest=1,
        beam_size=beam_size,
        lm_weight=lm_weight,
        word_score=word_score,
    )


def run_test(model, loader, device, decoder):
    model.eval()
    all_wer = []

    with torch.no_grad():
        for batch in loader:
            mels, input_lengths, targets, target_lengths = [b.to(device) for b in batch]
            logits = model(mels)
            log_probs = torch.log_softmax(logits, dim=-1).cpu()

            for j in range(mels.shape[0]):
                sample_log_probs = log_probs[j : j + 1, : input_lengths[j], :]
                results = decoder(sample_log_probs)

                if len(results[0]) > 0:
                    pred_tokens = results[0][0].tokens.tolist()
                    pred_text = int_to_text(pred_tokens).upper()
                else:
                    pred_text = ""

                start_idx = sum(target_lengths[:j]).item() if j > 0 else 0
                end_idx = start_idx + target_lengths[j].item()
                actual_text = int_to_text(targets[start_idx:end_idx].tolist()).upper()

                all_wer.append(wer(actual_text, pred_text))

    wer_avg = float(np.mean(all_wer))
    print(f"Average WER: {wer_avg:.4f}")
    return wer_avg


def main():
    scratch_path = "/scratch.hpc/marcello.spagnoli2/models/lm"
    os.makedirs(scratch_path, exist_ok=True)
    os.environ["TORCH_HOME"] = scratch_path
    files = download_pretrained_files("librispeech-3-gram")

    decoder_tokens = [c.lower() for c in CHARS]
    if " " in decoder_tokens:
        idx_spazio = decoder_tokens.index(" ")
        decoder_tokens[idx_spazio] = "|"

    test_ds = LibriSpeechDataset(DATA_DIR, "test-clean", download=DOWNLOAD_DATASET)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_audio_fn,
    )
    model = load_model(MODELS_DIR / "best_model.pth", DEVICE)

    def objective(trial):
        beam_size = trial.suggest_int("beam_size", 10, 50)
        lm_weight = trial.suggest_float("lm_weight", 0.1, 1.0)
        word_score = trial.suggest_float("word_score", -5.0, 5.0)

        decoder = build_decoder(
            beam_size, lm_weight, word_score, files=files, decoder_tokens=decoder_tokens
        )
        wer_avg = run_test(model, test_loader, DEVICE, decoder)
        return wer_avg

    n_trials = int(os.environ.get("N_TRIALS", 20))
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nBest decoder hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"Best WER: {study.best_value:.4f}")


if __name__ == "__main__":
    main()
