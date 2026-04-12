# Guida al Salvataggio e Caricamento dei Modelli

## 📁 Struttura dei File Salvati

Dopo l'addestramento, troverai i modelli nella directory `models/`:

```
models/
├── checkpoint_epoch_1.pth    # Checkpoint dopo l'epoca 1
├── checkpoint_epoch_2.pth    # Checkpoint dopo l'epoca 2
├── ...
├── checkpoint_epoch_N.pth    # Checkpoint dopo l'ultima epoca
├── best_model.pth            # 🌟 Miglior modello (test loss più bassa)
└── final_model.pth           # Modello finale dopo tutte le epoche
```

## 💾 Cosa Viene Salvato

Ogni file `.pth` contiene:
- `model_state_dict`: I pesi del modello
- `optimizer_state_dict`: Lo stato dell'ottimizzatore (per riprendere il training)
- `epoch`: Numero dell'epoca
- `train_loss`: Loss sul training set
- `test_loss`: Loss sul test set
- `model_config`: Configurazione del modello (solo in `final_model.pth`)

## 🔄 Come Caricare un Modello

### Opzione 1: Usa la funzione load_model

```python
from main import load_model, DEVICE, MODELS_DIR
from pathlib import Path

# Carica il miglior modello
model = load_model(MODELS_DIR / "best_model.pth", DEVICE)

# Oppure carica il modello finale
model = load_model(MODELS_DIR / "final_model.pth", DEVICE)

# Oppure carica un checkpoint specifico
model = load_model(MODELS_DIR / "checkpoint_epoch_3.pth", DEVICE)
```

### Opzione 2: Caricamento manuale

```python
import torch
from model import SpeechRecognitionModel
from text_utils import CHARS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il checkpoint
checkpoint = torch.load("models/best_model.pth", map_location=DEVICE)

# Crea il modello
model = SpeechRecognitionModel(n_mels=80, hidden_size=256, n_classes=len(CHARS))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

print(f"Modello caricato dall'epoca {checkpoint['epoch']}")
print(f"Test Loss: {checkpoint['test_loss']:.4f}")
```

## 🎯 Usare il Modello per Inferenza

```python
from main import load_model, run_test_inference, DEVICE, MODELS_DIR
from data_loader import LibriSpeechDataset, collate_audio_fn
from torch.utils.data import DataLoader
from pathlib import Path

# 1. Carica il modello
model = load_model(MODELS_DIR / "best_model.pth", DEVICE)

# 2. Prepara i dati
test_ds = LibriSpeechDataset(Path("./data"), "test-clean")
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_audio_fn)

# 3. Esegui l'inferenza
run_test_inference(model, test_loader, DEVICE, num_samples=10)
```

## 🔁 Riprendere l'Addestramento

Se vuoi continuare l'addestramento da un checkpoint:

```python
import torch
from torch import nn
from model import SpeechRecognitionModel
from text_utils import CHARS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica checkpoint
checkpoint = torch.load("models/checkpoint_epoch_3.pth", map_location=DEVICE)

# Ricrea modello e ottimizzatore
model = SpeechRecognitionModel(n_mels=80, hidden_size=256, n_classes=len(CHARS)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Carica i pesi salvati
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

print(f"Riprendo l'addestramento dall'epoca {start_epoch}")

# Continua il training loop...
```

## 🏆 Best Practices

1. **Usa `best_model.pth`** per l'inferenza in produzione (ha la test loss migliore)
2. **Salva `final_model.pth`** per avere un riferimento dell'ultimo stato
3. **Mantieni i checkpoint** se vuoi poter riprendere l'addestramento
4. **Monitora la test loss** durante il training per evitare overfitting

## 📊 Monitoraggio

Durante il training vedrai output come:

```
--- Epoca 3/5 ---
  Batch 0/100 | Loss: 45.2341
  ...
Fine Epoca 3:
  Train Loss Media: 42.5678
  Test Loss Media:  38.1234
  Checkpoint salvato: models/checkpoint_epoch_3.pth
  ⭐ Nuovo miglior modello salvato! (Test Loss: 38.1234)
```

La stella ⭐ indica che questo modello è il migliore finora!
