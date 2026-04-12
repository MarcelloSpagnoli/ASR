import torch
from torch import nn
from torch.utils.data import DataLoader
import random
import numpy as np

# Import dai moduli creati in precedenza
from data_loader import LibriSpeechDataset, collate_audio_fn
from text_utils import CHARS
from test import run_test_inference
from model import SpeechRecognitionModelV2
from train_evaluate import train_model
from configs import *

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    # 0. Set random seed for reproducibility
    set_seed(42)
    
    # 1. Crea la directory per salvare i modelli se non esiste
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Preparazione Dataset e DataLoader
    print("Caricamento dataset...")
    print(f"Directory dati: {DATA_DIR.absolute()}")
    print(f"Device: {DEVICE}")
    
    train_ds = LibriSpeechDataset(DATA_DIR, "train-clean-100", download=DOWNLOAD_DATASET)
    validation_ds = LibriSpeechDataset(DATA_DIR, "dev-clean", download=DOWNLOAD_DATASET)
    test_ds = LibriSpeechDataset(DATA_DIR, "test-clean", download=DOWNLOAD_DATASET)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_audio_fn
    )
    
    validation_loader = DataLoader(
        validation_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_audio_fn
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_audio_fn
    )

    # 2. Inizializzazione Modello
    # n_classes deve corrispondere alla lunghezza del dizionario CHARS
    model = SpeechRecognitionModelV2(n_mels=80, hidden_size=256, n_classes=len(CHARS), n_cnn_layers=3).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # blank=0 perché il '-' è in posizione 0 in CHARS
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # 3. Training Loop
    print(f"Inizio addestramento su: {DEVICE}\n")
    best_model_path = MODELS_DIR / "best_model.pth"
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=EPOCHS,
        device=DEVICE,
        save_path=best_model_path
    )
    
    # 4. Inferenza finale - carica il miglior modello
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    run_test_inference(model, test_loader, DEVICE)
    

if __name__ == "__main__":
    main()