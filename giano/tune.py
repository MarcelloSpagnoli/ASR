import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader

from configs import EPOCHS, DATA_DIR, DEVICE, NUM_WORKERS, DOWNLOAD_DATASET
from train_evaluate import train_model
from model import SpeechRecognitionModel
from data_loader import LibriSpeechDataset, collate_audio_fn


def objective(trial):
    """Objective function for Optuna to optimize."""
    torch.cuda.empty_cache()
    
    # Define hyperparameters to search
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_int('hidden_size', 128, 256, step=64)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    
    try:
        # Load datasets
        train_dataset = LibriSpeechDataset(DATA_DIR, "train-clean-100", download=DOWNLOAD_DATASET)
        val_dataset = LibriSpeechDataset(DATA_DIR, "dev-clean", download=DOWNLOAD_DATASET)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_audio_fn, num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_audio_fn, num_workers=NUM_WORKERS
        )
        
        # Create model and optimizer
        model = SpeechRecognitionModel(n_mels=80, hidden_size=hidden_size, n_classes=29).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CTCLoss(reduction='mean', zero_infinity=True)
        
        print(f"Trial {trial.number}: LR={learning_rate:.5f}, Hidden Size={hidden_size}, Batch Size={batch_size}")
        
        # Use unified train_model function with trial for pruning
        best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=EPOCHS,
            device=DEVICE,
            trial=trial
        )
        
        return best_val_loss
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ⚠ OOM with batch_size={batch_size}, hidden_size={hidden_size}")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        raise
    finally:
        torch.cuda.empty_cache()


def main():
    print(f"Starting hyperparameter tuning on {DEVICE}\n")
    
    # Create study and run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5, show_progress_bar=True, catch=(RuntimeError,))
    
    # Print results
    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETERS:")
    print("=" * 60)
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"\nBest validation loss: {study.best_value:.4f}")


if __name__ == "__main__":
    main()