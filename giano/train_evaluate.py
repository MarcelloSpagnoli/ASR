import torch


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        # Sposta i dati sul device (mels, input_lengths, targets, target_lengths)
        mels, input_lengths, targets, target_lengths = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        
        # Forward pass: l'output è [batch_size, time, n_classes]
        logits = model(mels)
        
        # CTCLoss si aspetta [T, B, C]
        log_probs = logits.permute(1, 0, 2)
        
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"--- Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
            
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            mels, in_lens, targets, tar_lens = [b.to(device) for b in batch]
            
            logits = model(mels)
            log_probs = logits.permute(1, 0, 2)
            
            loss = criterion(log_probs, targets, in_lens, tar_lens)
            total_loss += loss.item()
    
    return total_loss / len(loader)

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, save_path=None, trial=None):
    """
    Unified training function used by both main.py and tune.py.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        criterion: Loss criterion
        epochs: Number of epochs
        device: Device to train on
        save_path: Path to save best model (optional, for main.py)
        trial: Optuna trial object (optional, for tune.py pruning)
    
    Returns:
        best_val_loss: Best validation loss achieved
    """
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'validation_loss': val_loss,
                }, save_path)
                print(f"✓ Best model saved (Val Loss: {val_loss:.4f})")
        
        if trial:
            trial.report(val_loss, epoch - 1)
            if trial.should_prune():
                raise Exception("Trial pruned")
    
    return best_val_loss