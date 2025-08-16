import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """
    Pre-activation Residual MLP block:
      x -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm -> GELU -> Dropout -> Linear
      + optional shortcut projection if in/out dims differ
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, p: float = 0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout2 = nn.Dropout(p)

        # shortcut projection if needed
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

        # He init for linear layers
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        if isinstance(self.proj, nn.Linear):
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h = F.gelu(h)
        h = self.fc1(h)
        h = self.dropout1(h)

        h = self.ln2(h)
        h = F.gelu(h)
        h = self.fc2(h)
        h = self.dropout2(h)

        return h + self.proj(x)


class MLP(nn.Module):
    """
    Deeper residual MLP for tabular data.
    Interface is the same as before: MLP(in_dim) -> logits (shape [B]).
    """
    def __init__(self, in_dim: int):
        super().__init__()
        hidden1 = 256
        hidden2 = 256
        hidden3 = 128
        p = 0.2

        # Input dropout to regularize
        self.input_dropout = nn.Dropout(0.05)

        # Stack of residual blocks to improve gradient flow
        self.block1 = ResidualBlock(in_dim, hidden1, hidden2, p=p)
        self.block2 = ResidualBlock(hidden2, hidden2, hidden2, p=p)
        self.block3 = ResidualBlock(hidden2, hidden2, hidden3, p=p)

        # Final head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden3),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden3, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        out = self.head(x).squeeze(1)  # logits
        return out