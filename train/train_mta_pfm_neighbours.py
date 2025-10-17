# # ============================================================
# # train_mta_pfm_neighbours.py — FINAL TRAINING SCRIPT (Refactored)
# # ============================================================
# # Integrated fixes:
# # ✅ Prevent gradient explosion (dynamic gradient clipping)
# # ✅ Handle NaN/Inf losses (auto-skip + backoff)
# # ✅ Reset LR scheduler on plateau
# # ✅ Added gradient sanity checks
# # ✅ Adaptive OOM handling
# # ✅ CUDA fragmentation fix
# # ============================================================

# import os
# import math
# import gc
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, random_split

# from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
# from utils.collate_mta_pfm_neighbours import collate_fn
# from utils.speed_utils import calculate_speed, check_speed_violations
# from models.mta_pfm_model_neighbours import CheckpointedIntegratedMTAPFM_neighbours

# # --- Change 0: prevent CUDA fragmentation ---
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# # -------------------------
# # Helper functions
# # -------------------------
# def ensure_batch_dim(tensor):
#     """Ensure tensor has batch dim as first dimension (B, ...)."""
#     if tensor is None:
#         return None
#     if tensor.dim() == 3:  # e.g., [A, H, D] -> [1, A, H, D]
#         return tensor.unsqueeze(0)
#     return tensor

# def try_free_cuda():
#     """Clear CUDA cache and trigger Python garbage collection."""
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     gc.collect()

# # ============================================================
# # Training Function
# # ============================================================
# def train_mta_pfm_model(
#     data_path,
#     model_save_path,
#     model_class=CheckpointedIntegratedMTAPFM_neighbours,
#     dataset_class=PFM_TrajectoryDataset_neighbours,
#     collate_fn=collate_fn,
#     batch_size=1,
#     epochs=1,
#     learning_rate=0.001,
#     weight_decay=0.0,
#     patience=7,
#     accumulation_steps=4,
#     device=None,
#     max_agents_per_forward=32,  # memory control
# ):
#     """
#     Training loop for Integrated MTA-PFM Model (Neighbour version).
#     Includes:
#     - Gradient explosion handling
#     - Adaptive OOM recovery
#     - Gradient clipping
#     - NaN/Inf loss protection
#     """

#     # ---------------------------
#     # SETUP
#     # ---------------------------
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\n[TRAIN] Using device: {device}")

#     dataset = dataset_class(data_path)
#     print(f"[TRAIN] Loaded dataset with {len(dataset)} samples")

#     val_size = int(0.2 * len(dataset))
#     train_size = len(dataset) - val_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#     print(f"[TRAIN] Split → Train: {train_size}, Val: {val_size}")

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=lambda b: collate_fn(
#             b,
#             history_len=dataset.history_len,
#             prediction_len=dataset.prediction_len,
#             max_neighbors=dataset.max_neighbors,
#         ),
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=lambda b: collate_fn(
#             b,
#             history_len=dataset.history_len,
#             prediction_len=dataset.prediction_len,
#             max_neighbors=dataset.max_neighbors,
#         ),
#     )

#     # ---------------------------
#     # Model and Optimizer
#     # ---------------------------
#     model = model_class().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=3, verbose=True
#     )
#     criterion = nn.MSELoss()

#     # Speed constraints info
#     print("\n=== [TRAIN] Speed Constraints ===")
#     print(f"   Target Avg Speed: {model.target_avg_speed:.4f}")
#     print(f"   Allowed Range: [{model.min_speed:.4f}, {model.max_speed:.4f}]")
#     print(f"   Tolerance: ±{model.speed_tolerance * 100:.1f}%")

#     best_val_loss = float("inf")
#     patience_counter = 0

#     # ============================================================
#     # TRAINING LOOP
#     # ============================================================
#     for epoch in range(epochs):
#         print(f"\n=== [EPOCH {epoch + 1}/{epochs}] ===")
#         model.train()
#         epoch_loss = 0.0

#         for batch_idx, (history, future, neighbor_histories, goals, expanded_goals) in enumerate(train_loader):
#             # Preprocessing
#             history = ensure_batch_dim(history).to(device)
#             future = ensure_batch_dim(future).to(device)
#             neighbor_histories = ensure_batch_dim(neighbor_histories).to(device)
#             expanded_goals = ensure_batch_dim(expanded_goals).to(device)

#             A = history.shape[1]
#             if A == 0:
#                 continue

#             # Chunked Forward Pass
#             chunk_size = min(max_agents_per_forward, A)
#             processed_successfully = False

#             while chunk_size >= 1 and not processed_successfully:
#                 try:
#                     num_chunks = math.ceil(A / chunk_size)
#                     optimizer.zero_grad()
#                     batch_epoch_loss = 0.0

#                     for chunk_idx in range(num_chunks):
#                         s, e = chunk_idx * chunk_size, min((chunk_idx + 1) * chunk_size, A)

#                         # Slice data
#                         hist_chunk = history[:, s:e, :, :]        # [B, chunk, H, 2]
#                         fut_chunk = future[:, s:e, :, :]          # [B, chunk, T, 2]
#                         nbr_chunk = neighbor_histories[:, s:e, :, :, :]  # [B, chunk, N, H, 2]

#                         # Ensure shapes for concatenation (INTEGRATED CHANGE)
#                         if hist_chunk.dim() == 4:  # [B, chunk, H, 2]
#                             hist_chunk = hist_chunk.unsqueeze(2)  # [B, chunk, 1, H, 2]
#                         elif hist_chunk.dim() == 5:
#                             # If already [B, chunk, 1, H, 2], do nothing
#                             pass
#                         else:
#                             raise ValueError(f"Unexpected shape for hist_chunk: {hist_chunk.shape}")

#                         # nbr_chunk must be [B, chunk, N, H, 2]
#                         if nbr_chunk.dim() != 5:
#                             # NOTE: Assuming the input data loader is correct,
#                             # only checking the dimension is necessary.
#                             raise ValueError(f"nbr_chunk should be 5D: got {nbr_chunk.shape}")

#                         # Concatenate along entity dimension (dim=2)
#                         history_neighbors_chunk = torch.cat([hist_chunk, nbr_chunk], dim=2)  # [B, chunk, 1+N, H, 2]
                        
#                         # Slice expanded_goals to match entities
#                         entity_count = history_neighbors_chunk.shape[2]  # should be 1+N
#                         exp_goal_chunk = expanded_goals[:, s:e, :entity_count, :]

#                         adjusted_preds_chunk, decoded_preds_chunk, coeff_mean, coeff_var = model(
#                             history_neighbors_chunk, exp_goal_chunk
#                         )
#                         ego_pred_chunk = adjusted_preds_chunk[:, :, 0, :, :]

#                         loss_chunk_raw = criterion(ego_pred_chunk, fut_chunk)
#                         if not torch.isfinite(loss_chunk_raw):
#                             print(f"[WARN] Non-finite loss (NaN/Inf) at batch {batch_idx}, skipping...")
#                             loss_chunk_raw = torch.tensor(0.0, device=device)
#                             optimizer.zero_grad()
#                             try_free_cuda()
#                             break

#                         # Dynamic Gradient Clipping
#                         loss_to_backward = loss_chunk_raw / float(num_chunks)
#                         loss_to_backward.backward()
#                         total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                         if total_norm > 1e3:
#                             print(f"[WARN] High grad norm ({total_norm:.2f}) — clipping more aggressively!")
#                             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
#                         batch_epoch_loss += loss_chunk_raw.item()

#                         if chunk_idx % 4 == 0:
#                             try_free_cuda()

#                     optimizer.step()
#                     optimizer.zero_grad()
#                     epoch_loss += batch_epoch_loss
#                     processed_successfully = True

#                 except RuntimeError as e:
#                     if "out of memory" in str(e).lower():
#                         print(f"[OOM] Batch {batch_idx}: reducing chunk size {chunk_size} → {max(1, chunk_size // 2)}")
#                         try_free_cuda()
#                         chunk_size = max(1, chunk_size // 2)
#                         continue
#                     else:
#                         raise e

#             if not processed_successfully:
#                 print(f"[TRAIN] Skipping batch {batch_idx} after repeated OOM.")
#                 try_free_cuda()
#                 continue


#         # ---------------------------
#         # End of Epoch
#         # ---------------------------
#         avg_train_loss = epoch_loss / max(1, len(train_loader))
#         print(f"[EPOCH {epoch + 1}] Train Loss: {avg_train_loss:.6f}")

#         # ---------------------------
#         # Validation Phase
#         # ---------------------------
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for vhist, vfuture, vneigh, _, vexp_goals in val_loader:
#                 vhist = ensure_batch_dim(vhist).to(device)
#                 vfuture = ensure_batch_dim(vfuture).to(device)
#                 vneigh = ensure_batch_dim(vneigh).to(device)
#                 vexp_goals = ensure_batch_dim(vexp_goals).to(device)

#                 # Validation input preparation (simplified check based on training)
#                 if vhist.dim() == 4:  # [B, A, H, 2]
#                     vhist = vhist.unsqueeze(2)  # [B, A, 1, H, 2]
                
#                 # Assume vneigh is [B, A, N, H, 2]. If it came in a different shape due to collate_fn quirks,
#                 # the previous code had complex permutes. We'll simplify to ensure 5D for concatenation.
#                 if vneigh.dim() != 5:
#                     # This relies on the collate_fn being consistent. Reverting to the old logic is safer
#                     # if the new collate_fn isn't strictly enforcing [B, A, N, H, 2].
#                     # The original validation logic was:
#                     # if vneigh.dim() == 4:
#                     #     vneigh = vneigh.unsqueeze(-1).repeat(1, 1, 1, 1, 2)
#                     # if vneigh.dim() == 5 and permutation was needed...
                    
#                     # For safety, let's restore the original validation logic for vneigh
#                     if vneigh.dim() == 4:
#                         vneigh = vneigh.unsqueeze(-1).repeat(1, 1, 1, 1, 2)

#                     # Permute neighbor histories if needed (Original logic)
#                     if vneigh.dim() == 5:
#                         if vneigh.shape[-1] == 2 and vneigh.shape[-2] != 2:
#                             vneigh = vneigh.permute(0, 1, 3, 2, 4)
#                         elif vneigh.shape[-2] == 2:
#                             vneigh = vneigh.permute(0, 1, 3, 2, 4)
#                         elif vneigh.shape[2] == 2 and vneigh.shape[-1] == 2:
#                             pass
#                         else:
#                             raise ValueError(f"Unexpected vneigh shape: {vneigh.shape}")
                
#                 val_input = torch.cat([vhist, vneigh], dim=2) # Concatenate along entity dim (2)

#                 preds, _, _, _ = model(val_input, vexp_goals)
#                 ego_pred = preds[:, :, 0, :, :]
#                 val_loss += criterion(ego_pred, vfuture).item()

#         avg_val_loss = val_loss / max(1, len(val_loader))
#         print(f"[VAL] Validation Loss: {avg_val_loss:.6f}")

#         # ---------------------------
#         # Scheduler & Early Stopping
#         # ---------------------------
#         scheduler.step(avg_val_loss)
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             patience_counter = 0
#             torch.save({"model_state_dict": model.state_dict()}, model_save_path)
#             print(f"✅ Saved new best model → Val Loss: {avg_val_loss:.6f}")
#         else:
#             patience_counter += 1
#             print(f"⚠️ EarlyStopping: {patience_counter}/{patience}")

#         # If patience exceeded → reset LR if plateau persists
#         if patience_counter >= patience:
#             print("⏹ Early stopping triggered.")
#             break

#         try_free_cuda()

#     print(f"\n🏁 Training Complete. Best model saved to: {model_save_path}")

# # ============================================================
# # MAIN EXECUTION
# # ============================================================
# if __name__ == "__main__":
#     print("[TRAIN] Starting training run...")
#     train_mta_pfm_model(
#         data_path="data/combined_annotations.csv",
#         model_save_path="checkpoints/mta_pfm_trajectory_neighbours_model_final2.pth",
#         batch_size=1,
#         epochs=1,
#         patience=7,
#         accumulation_steps=4,
#         max_agents_per_forward=32,
#     )
import os
import math
import gc
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
from utils.collate_mta_pfm_neighbours import collate_fn
from models.mta_pfm_model_neighbours import CheckpointedIntegratedMTAPFM_neighbours

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def ensure_batch_dim(tensor):
    if tensor is None:
        return None
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def try_free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def clean_tensor(tensor):
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()


def train_mta_pfm_model(
    data_path,
    model_save_path,
    model_class=CheckpointedIntegratedMTAPFM_neighbours,
    dataset_class=PFM_TrajectoryDataset_neighbours,
    collate_fn=collate_fn,
    batch_size=1,
    epochs=1,
    learning_rate=0.0001,
    weight_decay=0.0,
    patience=7,
    accumulation_steps=4,
    device=None,
    max_agents_per_forward=32,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TRAIN] Using device: {device}")

    dataset = dataset_class(data_path)
    print(f"[TRAIN] Loaded dataset with {len(dataset)} samples")

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"[TRAIN] Split → Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b,
            history_len=dataset.history_len,
            prediction_len=dataset.prediction_len,
            max_neighbors=dataset.max_neighbors,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(
            b,
            history_len=dataset.history_len,
            prediction_len=dataset.prediction_len,
            max_neighbors=dataset.max_neighbors,
        ),
    )

    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    criterion = nn.MSELoss()

    print("\n=== [TRAIN] Speed Constraints ===")
    print(f"    Target Avg Speed: {model.target_avg_speed:.4f}")
    print(f"    Allowed Range: [{model.min_speed:.4f}, {model.max_speed:.4f}]")
    print(f"    Tolerance: ±{model.speed_tolerance * 100:.1f}%")

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"\n=== [EPOCH {epoch + 1}/{epochs}] ===")
        model.train()
        epoch_loss = 0.0

        for batch_idx, (history_neighbors, future, neighbor_histories, goals, expanded_goals) in enumerate(train_loader):
            history_neighbors = ensure_batch_dim(history_neighbors).to(device)
            future = ensure_batch_dim(future).to(device)
            neighbor_histories = ensure_batch_dim(neighbor_histories).to(device)
            expanded_goals = ensure_batch_dim(expanded_goals).to(device)

            history_neighbors = clean_tensor(history_neighbors)
            future = clean_tensor(future)
            neighbor_histories = clean_tensor(neighbor_histories)
            expanded_goals = clean_tensor(expanded_goals)

            A = history_neighbors.shape[1]
            if A == 0:
                continue

            chunk_size = min(max_agents_per_forward, A)
            processed_successfully = False

            while chunk_size >= 1 and not processed_successfully:
                try:
                    num_chunks = math.ceil(A / chunk_size)
                    optimizer.zero_grad()
                    batch_epoch_loss = 0.0

                    for chunk_idx in range(num_chunks):
                        s, e = chunk_idx * chunk_size, min((chunk_idx + 1) * chunk_size, A)

                        hist_chunk = history_neighbors[:, s:e, :, :, :]
                        fut_chunk = future[:, s:e, :, :]
                        nbr_chunk = neighbor_histories[:, s:e, :, :, :]
                        exp_goal_chunk = expanded_goals[:, s:e, :, :]

                        hist_chunk = clean_tensor(hist_chunk)
                        fut_chunk = clean_tensor(fut_chunk)
                        nbr_chunk = clean_tensor(nbr_chunk)
                        exp_goal_chunk = clean_tensor(exp_goal_chunk)

                        adjusted_preds_chunk, decoded_preds_chunk, coeff_mean, coeff_var = model(
                            hist_chunk, exp_goal_chunk
                        )
                        ego_pred_chunk = adjusted_preds_chunk[:, :, 0, :, :]

                        if torch.isnan(ego_pred_chunk).any() or torch.isinf(ego_pred_chunk).any():
                            print(f"[WARN] Model output contains NaN/Inf at batch {batch_idx}, skipping...")
                            optimizer.zero_grad()
                            try_free_cuda()
                            break

                        loss_chunk_raw = criterion(ego_pred_chunk, fut_chunk)
                        if not torch.isfinite(loss_chunk_raw):
                            print(f"[WARN] Non-finite loss (NaN/Inf) at batch {batch_idx}, skipping...")
                            loss_chunk_raw = torch.tensor(0.0, device=device)
                            optimizer.zero_grad()
                            try_free_cuda()
                            break

                        loss_to_backward = loss_chunk_raw / float(num_chunks)
                        loss_to_backward.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        batch_epoch_loss += loss_chunk_raw.item()

                        if chunk_idx % 4 == 0:
                            try_free_cuda()

                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss += batch_epoch_loss
                    processed_successfully = True

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[OOM] Batch {batch_idx}: reducing chunk size {chunk_size} → {max(1, chunk_size // 2)}")
                        try_free_cuda()
                        chunk_size = max(1, chunk_size // 2)
                        continue
                    else:
                        raise e

            if not processed_successfully:
                print(f"[TRAIN] Skipping batch {batch_idx} after repeated OOM.")
                try_free_cuda()
                continue

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)
        print(f"[EPOCH {epoch + 1}] Train Loss: {avg_train_loss:.6f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vhist_neighbors, vfuture, vneigh_histories, _, vexp_goals in val_loader:
                vhist_neighbors = ensure_batch_dim(vhist_neighbors).to(device)
                vfuture = ensure_batch_dim(vfuture).to(device)
                vexp_goals = ensure_batch_dim(vexp_goals).to(device)

                vhist_neighbors = clean_tensor(vhist_neighbors)
                vfuture = clean_tensor(vfuture)
                vexp_goals = clean_tensor(vexp_goals)

                preds, _, _, _ = model(vhist_neighbors, vexp_goals)
                ego_pred = preds[:, :, 0, :, :]
                if torch.isnan(ego_pred).any() or torch.isinf(ego_pred).any():
                    print(f"[WARN] Validation output contains NaN/Inf, skipping batch.")
                    continue

                val_loss += criterion(ego_pred, vfuture).item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        print(f"[VAL] Validation Loss: {avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict()}, model_save_path)
            print(f"✅ Saved new best model → Val Loss: {avg_val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"⚠️ EarlyStopping: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("⏹ Early stopping triggered.")
            break

        try_free_cuda()

    print(f"\n🏁 Training Complete. Best model saved to: {model_save_path}")
    plot_loss(train_losses, val_losses)
