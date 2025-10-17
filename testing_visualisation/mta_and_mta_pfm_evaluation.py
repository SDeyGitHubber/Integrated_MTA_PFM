import torch
from torch.utils.data import DataLoader

# === Dataset Import ===
from datasets.pfm_trajectory_dataset import PFM_TrajectoryDataset

# === Models ===
from models.deep_lstm_model import DeepLSTMModel
from models.mta_pfm_model import CheckpointedIntegratedMTAPFMModel


# === Metrics ===
def compute_ADE(pred, gt):
    return torch.norm(pred - gt, dim=-1).mean().item()

def compute_FDE(pred, gt):
    return torch.norm(pred[:, -1, :] - gt[:, -1, :], dim=-1).mean().item()

def compute_miss_rate(pred, gt, threshold=2.0):
    dist = torch.norm(pred[:, -1, :] - gt[:, -1, :], dim=-1)
    misses = (dist > threshold).float()
    return misses.mean().item()


# === Unified Testing Function ===
def test_with_metrics(model_path, cleaned_txt_file):
    """
    Evaluate trajectory prediction for DeepLSTM or Integrated MTA+PFM models.
    Args:
        model_path (str): Path to model checkpoint (.pth file).
        cleaned_txt_file (str): Path to preprocessed dataset file.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Dataset ---
    dataset = PFM_TrajectoryDataset(cleaned_txt_file)
    print(f"\n Loaded {len(dataset)} frame samples for evaluation.")

    if len(dataset) == 0:
        print(" No valid samples available. Check preprocessing.")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- Model selection ---
    model_path_lower = model_path.lower()
    if "deep_lstm" in model_path_lower:
        model = DeepLSTMModel().to(device)
        model_name = "DeepLSTM Baseline"
    elif "integrated" in model_path_lower or "mta_pfm" in model_path_lower:
        model = CheckpointedIntegratedMTAPFMModel().to(device)
        model_name = "Integrated MTA+PFM Baseline"
    else:
        raise ValueError(f" Unknown model type in path: {model_path}")

    print(f"\n Loading model: {model_name}")

    # --- Load checkpoint ---
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Metrics aggregation ---
    total_ade, total_fde, total_miss, count = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for history, future, neighbors, goal in dataloader:
            history, future, neighbors, goal = (
                history.to(device), future.to(device), neighbors.to(device), goal.to(device)
            )

            pred, _, _ = model(history, neighbors, goal)  # [1, A, T, 2]
            pred, future = pred[0], future[0]             # [A, T, 2]

            # Fix length mismatch
            min_len = min(pred.size(1), future.size(1))
            if pred.size(1) != future.size(1):
                print(f" Truncating: pred_len={pred.size(1)}, future_len={future.size(1)} → using {min_len}")
            pred, future = pred[:, :min_len, :], future[:, :min_len, :]

            # Metrics
            ade = compute_ADE(pred, future)
            fde = compute_FDE(pred, future)
            miss = compute_miss_rate(pred, future)

            total_ade += ade
            total_fde += fde
            total_miss += miss
            count += 1

    if count == 0:
        print(" Evaluation aborted: No samples processed.")
        return

    # --- Final Report ---
    print(f"\n Evaluation Metrics on Test Dataset ({model_name}):")
    print(f"🔹 Average ADE:  {total_ade / count:.4f}")
    print(f"🔹 Average FDE:  {total_fde / count:.4f}")
    print(f"🔹 Miss Rate:    {total_miss / count:.4f} (threshold: 2m)")


# === Example Usage ===
if __name__ == "__main__":
    test_with_metrics(
        "/home/dasari-raj-vamsi/Desktop/iDataHub_running/checkpoints/integrated_pfm_model.pth",
        "/home/dasari-raj-vamsi/Desktop/iDataHub_running/data/combined_annotations.csv"
    )

    test_with_metrics(
        "/home/dasari-raj-vamsi/Desktop/iDataHub_running/checkpoints/deep_lstm_model.pth",
        "/home/dasari-raj-vamsi/Desktop/iDataHub_running/data/combined_annotations.csv"
    )