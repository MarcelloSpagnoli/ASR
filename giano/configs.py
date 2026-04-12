import os
from pathlib import Path
import torch

BATCH_SIZE =  32
LR =  0.0007697696144499126
EPOCHS = 15
DATA_DIR = Path("/scratch.hpc/marcello.spagnoli2/data") 
MODELS_DIR = Path("/scratch.hpc/marcello.spagnoli2/models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
CNN_LAYERS_V2 = 3

DOWNLOAD_DATASET = False 