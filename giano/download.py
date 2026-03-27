
"""Download LibriSpeech"""

import torchaudio
import sys
from tqdm import tqdm
from pathlib import Path

SPLIT = "train-clean-100"
DATA_DIR = Path("./data")


DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset:   {SPLIT}")
print(f"Directory: {DATA_DIR.resolve()}")
print()

try:
    print(f"Download in corso...")
    ds = torchaudio.datasets.LIBRISPEECH(
        str(DATA_DIR),
        url=SPLIT,
        download=True,
    )
except Exception as e:
    print(f"Errore durante il download: {e}", file=sys.stderr)
    sys.exit(1)

print(f"\nDownload completato! {len(ds)} campioni trovati.")
