import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.pfm_trajectory_dataset import PFM_TrajectoryDataset
from models.pfm_model import PFMOnlyModel
from utils.collate import collate_fn
from models.pfm_no_learnable import PFMOnlyModelNoLearnable

# === Metrics ===
def compute_ADE(pred, gt):
    return F.mse_loss(pred, gt, reduction='none').sum(dim=2).sqrt().mean().item()

def compute_FDE(pred, gt):
    return F.mse_loss(pred[:, -1], gt[:, -1], reduction='none').sum(dim=1).sqrt().mean().item()

def compute_miss_rate(pred, gt, threshold=2.0):
    final_dist = F.mse_loss(pred[:, -1], gt[:, -1], reduction='none').sum(dim=1).sqrt()
    misses = (final_dist > threshold).float()
    return (misses.sum() / len(misses)).item()


# === Trainable PFM Model Evaluation ===
def test_with_metrics_pfm(model_path, data_path, batch_size=1):
    """
    Evaluate a TRAINABLE PFM trajectory prediction model on a dataset.
    
    Args:
        model_path (str): Path to the saved model .pth file
        data_path (str): Path to the cleaned dataset file
        batch_size (int): Batch size for DataLoader (default=1)
    """
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PFM_TrajectoryDataset(data_path)
    print(f"\n Loaded {len(dataset)} frame samples for evaluation from {data_path}.")

    if len(dataset) == 0:
        print(" No valid samples available. Check your preprocessing.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = PFMOnlyModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_ade, total_fde, total_miss = 0.0, 0.0, 0.0
    count = 0

    with torch.no_grad():
        for history, future, neighbors, goal in dataloader:
            history, future, neighbors, goal = history.to(device), future.to(device), neighbors.to(device), goal.to(device)

            pred, _, _ = model(history, neighbors, goal)  # [B, A, T, 2]
            pred, future = pred[0], future[0]            # [A, T, 2]

            # Ensure lengths match
            min_len = min(pred.size(1), future.size(1))
            if pred.size(1) != future.size(1):
                print(f" Truncating: pred_len={pred.size(1)}, future_len={future.size(1)} → using {min_len}")
            pred, future = pred[:, :min_len, :], future[:, :min_len, :]

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

    print(f"\n Trainable PFM Evaluation Metrics on Test Dataset:")
    print(f"🔹 Average ADE:  {total_ade / count:.4f}")
    print(f"🔹 Average FDE:  {total_fde / count:.4f}")
    print(f"🔹 Miss Rate:    {total_miss / count:.4f} (threshold: 2m)")


# === Non-Trainable PFM Model Evaluation ===
def test_with_metrics_pfm_nontrainable(data_path, batch_size=1):
    """
    Evaluate a NON-TRAINABLE PFM trajectory prediction model on a dataset.
    
    Args:
        data_path (str): Path to the cleaned dataset file
        batch_size (int): Batch size for DataLoader (default=1)
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PFM_TrajectoryDataset(data_path)
    print(f"\n Loaded {len(dataset)} frame samples for evaluation from {data_path}.")

    if len(dataset) == 0:
        print(" No valid samples available. Check your preprocessing.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = PFMOnlyModelNoLearnable().to(device)
    model.eval()

    total_ade, total_fde, total_miss = 0.0, 0.0, 0.0
    count = 0

    with torch.no_grad():
        for history, future, neighbors, goal in dataloader:
            history, future, neighbors, goal = history.to(device), future.to(device), neighbors.to(device), goal.to(device)

            pred, _, _ = model(history, neighbors, goal)  # [B, A, T, 2]
            pred, future = pred[0], future[0]            # [A, T, 2]

            # Ensure lengths match
            min_len = min(pred.size(1), future.size(1))
            if pred.size(1) != future.size(1):
                print(f"⚠️ Truncating: pred_len={pred.size(1)}, future_len={future.size(1)} → using {min_len}")
            pred, future = pred[:, :min_len, :], future[:, :min_len, :]

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

    print(f"\n Non-Trainable PFM Evaluation Metrics on Test Dataset:")
    print(f"🔹 Average ADE:  {total_ade / count:.4f}")
    print(f"🔹 Average FDE:  {total_fde / count:.4f}")
    print(f"🔹 Miss Rate:    {total_miss / count:.4f} (threshold: 2m)")


# === Example Usage ===
if __name__ == "__main__":
    # Trainable PFM Model
    test_with_metrics_pfm(
        "/home/dasari-raj-vamsi/Desktop/iDataHub_running/checkpoints/pfm_only_model.pth",
        "/home/dasari-raj-vamsi/Desktop/iDataHub_running/data/combined_annotations.csv"
    )

    # Non-Trainable PFM Model
    test_with_metrics_pfm_nontrainable(
        "/home/dasari-raj-vamsi/Desktop/iDataHub_running/data/combined_annotations.csv"
    )
