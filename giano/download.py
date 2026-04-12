
"""Download LibriSpeech"""

import torchaudio
import sys
from tqdm import tqdm
from pathlib import Path

SPLIT_TRAINING = "train-clean-100"
SPLIT_VALIDATION = "dev-clean"
SPLIT_TEST = "test-clean"
file = "DOWNLOAD: "
splits = [SPLIT_TRAINING, SPLIT_VALIDATION, SPLIT_TEST]

DATA_DIR = Path("./data/LibriSpeech")
DATA_DIR.mkdir(parents=True, exist_ok=True)

for split in splits:
    if (DATA_DIR / split).exists():
        print(f"{file}Il dataset '{split}' è già presente in '{DATA_DIR.resolve()}'.")
        continue
    print(f"{file}Dataset: {split}")
    print(f"{file}Directory: {DATA_DIR.resolve()}")
    print()

    try:
        ds = torchaudio.datasets.LIBRISPEECH(
            str(DATA_DIR),
            url=split,
            download=True,
        )
    except Exception as e:
        print(f"{file}Errore durante il download: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"{file}\n{len(ds)} campioni trovati.")

