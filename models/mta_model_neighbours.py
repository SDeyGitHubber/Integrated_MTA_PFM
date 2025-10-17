import torch.nn as nn
import torch.utils.checkpoint as cp
import torch

class CheckpointedIntegratedMTAModelNeighbours(nn.Module):
    """
    LSTM-based multi-agent trajectory prediction model (without Potential Field).
    Uses an LSTM encoder/decoder for trajectory forecasting and projects interaction coefficients.
    """

    def __init__(self, input_size=2, hidden_size=64, num_layers=2,
                 target_avg_speed=4.087, speed_tolerance=0.15, num_agents=1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_embed = nn.Linear(input_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.coeff_projector = nn.Linear(hidden_size, 3)
        self.output = nn.Linear(hidden_size, input_size)
        if target_avg_speed is None:
            raise ValueError("target_avg_speed must be provided")
        self.target_avg_speed = target_avg_speed
        self.speed_tolerance = speed_tolerance
        self.min_speed = target_avg_speed * (1 - speed_tolerance)
        self.max_speed = target_avg_speed * (1 + speed_tolerance)
        self.dt = 0.1

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_decoder(self, x, hx):
        return self.decoder(x, hx)

    def forward(self, history_neighbors, goal):
        """
        Main forward pass for trajectory forecasting.
        Args:
            history_neighbors (Tensor): [B, A, ent, H, 2]
            goal (Tensor): [B, A, ent, 2] or [B, A, 2]
        Returns:
            decoded_preds (Tensor), coeff_mean (Tensor), coeff_var (Tensor)
        """
        B, A, ent, H, D = history_neighbors.shape
        device = history_neighbors.device

        if goal.shape == (B, A, D):
            goal = goal.unsqueeze(2).expand(B, A, ent, D).contiguous()
        elif goal.shape == (B, A, ent, D):
            pass
        else:
            raise AssertionError(f"Goal shape mismatch after expansion, got {goal.shape}, expected {(B, A, ent, D)}")

        hist_flat = history_neighbors.reshape(B * A * ent, H, D)
        emb = self.input_embed(hist_flat)
        _, (h_n, c_n) = cp.checkpoint(self.forward_encoder, emb)

        h_top = h_n[-1].reshape(B, A, ent, self.hidden_size)
        coeffs = self.coeff_projector(h_top)

        pred_len = 12
        pred_flat = torch.zeros(B * A * ent, pred_len, D, device=device)
        hx = (h_n, c_n)

        history_flat = history_neighbors.view(B * A * ent, H, D)
        last_pos = torch.zeros(B * A * ent, 1, D, device=device)
        for idx in range(B * A * ent):
            for t in range(H - 1, -1, -1):
                pos = history_flat[idx, t]
                if not torch.all(pos == 0):
                    last_pos[idx] = pos
                    break
            else:
                last_pos[idx] = torch.zeros(D, device=device)

        for t in range(pred_len):
            if t == 0:
                decoder_in = last_pos.clone()
            else:
                decoder_in = pred_flat[:, t - 1:t].clone()
            dec_emb = self.input_embed(decoder_in)
            h, c = hx
            hx = (h.contiguous(), c.contiguous())
            out, hx = cp.checkpoint(self.forward_decoder, dec_emb, hx)
            step_out = self.output(out.squeeze(1))
            if t == 0:
                pred_flat[:, t] = last_pos.squeeze(1) + step_out
            else:
                pred_flat[:, t] = pred_flat[:, t - 1] + step_out

        decoded_preds = pred_flat.reshape(B, A, ent, pred_len, D)
        coeff_mean = coeffs.mean()
        coeff_var = coeffs.var(unbiased=False)
        # No physical correction, just return decoded_preds
        return decoded_preds, decoded_preds, coeff_mean, coeff_var