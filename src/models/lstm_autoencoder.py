"""PyTorch LSTM Autoencoder for temporal sequence anomaly detection.

Implements a sequence-to-sequence reconstruction autoencoder trained on nominal
telemetry patterns to flag non-stationary temporal drift and thermal oscillations.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoderNet(nn.Module):
    """Sequence-to-sequence LSTM autoencoder network."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        seq_len: int = 6,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        # Encoder: compresses sliding sequence window into latent bottleneck
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder: reconstructs sequence from latent bottleneck
        self.latent_to_dec = nn.Linear(latent_dim, latent_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass reconstructing input sequence.

        Args:
            x: Input sequence tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tuple of (reconstructed sequence of same shape, latent vector z).
        """
        # Encoder pass
        _, (hidden, _) = self.encoder_lstm(x)
        # Last hidden state of encoder
        last_hidden = hidden[-1]  # shape: (batch_size, hidden_dim)
        z = torch.relu(self.enc_to_latent(last_hidden))  # shape: (batch_size, latent_dim)

        # Repeat latent vector across sequence length (RepeatVector)
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch_size, seq_len, latent_dim)
        z_dec = self.latent_to_dec(z_repeated)

        # Decoder pass
        dec_out, _ = self.decoder_lstm(z_dec)  # (batch_size, seq_len, hidden_dim)
        reconstructed = self.output_linear(dec_out)  # (batch_size, seq_len, input_dim)

        return reconstructed, z


class LSTMAutoencoderDetector:
    """Wrapper managing training, evaluation, and score calibration for LSTM Autoencoder."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        default_params: Dict[str, Any] = {
            "input_dim": 6,
            "seq_len": 6,
            "hidden_dim": 32,
            "latent_dim": 16,
            "num_layers": 1,
            "dropout": 0.1,
            "batch_size": 64,
            "learning_rate": 0.003,
            "epochs": 8,
            "weight_decay": 0.0001,
            "reconstruction_loss_percentile": 88.0,
            "gradient_clip_val": 1.0,
            "device": "cpu",
        }
        if config:
            default_params.update(config)
        default_params.update(kwargs)

        self.params = default_params
        self.device = torch.device(self.params.get("device", "cpu"))
        self.model = LSTMAutoencoderNet(
            input_dim=self.params["input_dim"],
            hidden_dim=self.params["hidden_dim"],
            latent_dim=self.params["latent_dim"],
            seq_len=self.params["seq_len"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
        ).to(self.device)

        self.threshold = 1.0
        self.norm_scale = 1.0
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, X_nominal_seqs: np.ndarray) -> "LSTMAutoencoderDetector":
        """Train autoencoder strictly on standardized nominal sequences."""
        self.feat_mean = np.mean(X_nominal_seqs, axis=(0, 1), keepdims=True)
        self.feat_std = np.std(X_nominal_seqs, axis=(0, 1), keepdims=True) + 1e-4
        norm_seqs = (X_nominal_seqs - self.feat_mean) / self.feat_std

        self.model.train()
        tensor_data = torch.tensor(norm_seqs, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )
        criterion = nn.MSELoss(reduction="mean")

        for _ in range(self.params["epochs"]):
            for (batch,) in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                reconstructed, _ = self.model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(self.params["gradient_clip_val"]),
                )
                optimizer.step()

        # Compute calibration threshold on nominal training set
        errors = self.compute_reconstruction_errors(X_nominal_seqs)
        pct = float(self.params["reconstruction_loss_percentile"])
        self.threshold = float(np.percentile(errors, pct))
        self.norm_scale = max(float(np.percentile(errors, 98.0)), 1e-4)
        self.is_fitted = True
        return self

    def compute_reconstruction_errors(self, X_seqs: np.ndarray) -> np.ndarray:
        """Compute MSE reconstruction error per sequence with standardization."""
        self.model.eval()
        arr = np.array(X_seqs, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)

        if self.feat_mean is not None and self.feat_std is not None:
            arr = (arr - self.feat_mean) / self.feat_std

        tensor_data = torch.tensor(arr, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            reconstructed, _ = self.model(tensor_data)
            squared_diff = (reconstructed - tensor_data) ** 2
            errors = squared_diff.mean(dim=(1, 2)).cpu().numpy()

        return errors

    def predict_score(self, X_seqs: np.ndarray) -> np.ndarray:
        """Return normalized temporal reconstruction anomaly score s_lstm(x) in [0, 1]."""
        errors = self.compute_reconstruction_errors(X_seqs)
        scaled = errors / self.norm_scale
        scores = 1.0 / (1.0 + np.exp(-3.0 * (scaled - 1.0)))
        return np.clip(scores, 0.0, 1.0)
