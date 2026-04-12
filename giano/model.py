import torch
from torch import nn
import torch.nn.functional as F
from text_utils import CHARS
from configs import CNN_LAYERS_V2

class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_mels=80, hidden_size=256, n_classes=29):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        # n_mels // 4 perché abbiamo due MaxPool2d sulla dimensione frequenza
        self.gru = nn.GRU(
            input_size=64 * (n_mels // 4),
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True # Aggiunto bidirezionale per performance migliori
        )

        self.linear = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1) # [B, 1, T, F]
        x = self.conv_block(x) # [B, 64, T, F//4]
        
        B, C, T, F_conv = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F_conv)
        
        x, _ = self.gru(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)

class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x  # [B, C, T, F]
        x = self.bn1(x)
        x = F.gelu(x) # GELU spesso performa meglio di ReLU in ASR
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x += residual # Connessione residua
        return x

class SpeechRecognitionModelV2(nn.Module):
    def __init__(self, n_mels=80, hidden_size=512, n_classes=29):
        super().__init__()
        
        # Pre-convoluzione per proiettare i canali
        self.conv_in = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        
        # Blocchi residui
        self.res_cnn = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=0.1, n_feats=n_mels) 
            for _ in range(CNN_LAYERS_V2)
        ])
        
        # Proiezione per la GRU
        self.fully_connected = nn.Linear(32 * n_mels, hidden_size)
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=3, # Più layer per catturare gerarchie complesse
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.linear = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        # x: [B, T, F] -> [B, 1, T, F]
        x = x.unsqueeze(1)
        x = self.conv_in(x)
        x = self.res_cnn(x)
        
        # Reshape per la parte ricorrente
        B, C, T, F_conv = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, C * F_conv)
        
        x = self.fully_connected(x)
        x, _ = self.gru(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)
    

def load_model(checkpoint_path, device):
    """Carica un modello salvato da un checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Crea il modello con la configurazione salvata (o usa i valori di default)
    if 'model_config' in checkpoint:
        print("Caricamento modello con configurazione salvata...")
        config = checkpoint['model_config']
        model = SpeechRecognitionModelV2(
            n_mels=config['n_mels'],
            hidden_size=config['hidden_size'],
            n_classes=config['n_classes']
        ).to(device)
    else:
        # Fallback con valori di default
        model = SpeechRecognitionModelV2(n_mels=80, hidden_size=256, n_classes=len(CHARS)).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Modello caricato da: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"   Epoca: {checkpoint['epoch']}")
    if 'validation_loss' in checkpoint:
        print(f"   Validation Loss: {checkpoint['validation_loss']:.4f}")
    
    return model