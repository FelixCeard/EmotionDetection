# define a 1-D CNN model

# import torch
import torch.nn as nn
# import torch.nn.functional as F
# from encodec import EncodecModel
import torch

class QuantizationSimulator(nn.Module):
    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits

    def forward(self, x):
        # Quantize and dequantize x to the specified number of bits, but let gradients flow as if identity
        if self.bits >= 32:
            return x
        qmin = 0
        qmax = 2 ** self.bits - 1
        x_min = x.detach().min()
        x_max = x.detach().max()
        # Avoid division by zero
        scale = (x_max - x_min) / float(qmax - qmin) if (x_max - x_min) != 0 else 1.0

        # Quantize
        x_int = torch.clamp(torch.round((x - x_min) / scale), qmin, qmax)
        # Dequantize
        x_quant = x_int * scale + x_min

        # Straight-through estimator: gradients flow as if identity
        return x + (x_quant - x).detach()
    


class Model(nn.Module):

    def __init__(self, num_mfccs_features=20, num_classes=8, dropout_rate=0.33):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=num_mfccs_features, out_channels=32, kernel_size=7, stride=3),
            QuantizationSimulator(bits=8),
            # nn.MaxPool1d(kernel_size=5, stride=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            QuantizationSimulator(bits=8),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            QuantizationSimulator(bits=8),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            QuantizationSimulator(bits=8),
            nn.BatchNorm1d(64),
            nn.ELU(),
            # nn.Dropout(dropout_rate),
            # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            # QuantizationSimulator(bits=8),
            # nn.BatchNorm1d(64),
            # nn.ELU(),
            # nn.Dropout(dropout_rate),
            # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            # QuantizationSimulator(bits=8),
            # nn.BatchNorm1d(64),
            # nn.ELU(),
            # nn.Dropout(dropout_rate),
            # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            # QuantizationSimulator(bits=8),
            # nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            QuantizationSimulator(bits=8),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            # QuantizationSimulator(bits=8),
            # nn.BatchNorm1d(64),
            # nn.ELU()
        )

        self.classifier = nn.Sequential(
            # nn.Linear(in_features=64, out_features=32),
            # nn.ELU(),
            nn.Linear(in_features=64, out_features=num_classes)
        )
    def forward(self, x):
        # print(x.shape)
        x = self.feature_extractor(x)
        # print(x.shape)
        x = x.mean(dim=2)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x