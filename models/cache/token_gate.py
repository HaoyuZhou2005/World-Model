"""
Token-level adaptive gate.

TODO:
1. Connect with Wan2.1 transformer hidden states.
2. Tune architecture and training objective.
3. Support hard gate inference.
"""

import torch
import torch.nn as nn


class TokenGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_state):
        """
        Args:
            hidden_state:
                [B, N, C]
                N: visual tokens

        Returns:
            gate:
                [B, N, 1]

        TODO:
            Connect gate with cache controller.
        """
        return self.mlp(hidden_state)
