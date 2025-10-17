import torch
import numpy as np
import matplotlib.pyplot as plt
from models.pfm_no_learnable import PFMOnlyModelNoLearnable

# =============================
# 2. Replace pred_function usage
# =============================
import matplotlib.pyplot as plt
import numpy as np

import torch
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(dataset, model, device, sample_idx=0, prev=12, next=12, max_neighbors=12):
    """
    Visualize trajectories for one frame/sample:
      - Past (history)
      - Ground-truth future
      - Predicted future (PFMOnlyModelNoLearnable)
      - Up to 12 neighbors (instantaneous positions)

    Args:
        dataset: object whose __getitem__ returns (history, future, neighbors, goals)
                 history:   [A, H, 2]
                 future:    [A, T, 2]
                 neighbors: [A, N, 2]   (instantaneous neighbor positions)
                 goals:     [A, 2]
        model: PFMOnlyModelNoLearnable (already constructed)
        device: torch.device
        sample_idx: int, sample/frame index
        prev: number of observed steps to plot from history (clamped to H)
        next: number of future steps to plot from GT/preds (clamped to T/12)
        max_neighbors: how many neighbors to draw per agent
    """
    model.eval()

    # ---- 1) Get sample and move to device ----
    sample = dataset[sample_idx]  # should be 4-tuple
    if not isinstance(sample, (tuple, list)) or len(sample) != 4:
        raise ValueError(
            f"Expected dataset[sample_idx] -> (history, future, neighbors, goals) "
            f"but got type {type(sample)} with length {len(sample) if hasattr(sample,'__len__') else 'N/A'}"
        )
    history, future, neighbors, goals = sample  # [A,H,2], [A,T,2], [A,N,2], [A,2]

    if history.numel() == 0:
        print(f"Sample {sample_idx}: no agents to plot.")
        return

    # Clamp lengths for display only (model always predicts 12 steps)
    H = history.shape[1]
    T = future.shape[1]
    prev = min(prev, H)
    next = min(next, 12, T)  # model outputs 12; GT may be shorter/longer

    # ---- 2) Prepare batched tensors for the model ----
    history_b  = history.unsqueeze(0).to(device)   # [1, A, H, 2]
    neighbors_b= neighbors.unsqueeze(0).to(device) # [1, A, N, 2]
    goals_b    = goals.unsqueeze(0).to(device)     # [1, A, 2]

    with torch.no_grad():
        preds_b, _, _ = model(history_b, neighbors_b, goals_b)  # [1, A, 12, 2]
    preds = preds_b.squeeze(0).cpu()                             # [A, 12, 2]

    history_np  = history.cpu().numpy()                          # [A, H, 2]
    future_np   = future.cpu().numpy()                           # [A, T, 2]
    preds_np    = preds.numpy()                                  # [A, 12, 2]
    neighbors_np= neighbors.cpu().numpy()                        # [A, N, 2]

    A = history_np.shape[0]
    cmap = plt.cm.get_cmap('tab10', A)

    # ---- 3) Plot each agent with up to 12 nearest neighbors (instant positions) ----
    for ego in range(A):
        plt.figure(figsize=(9, 8))

        # Past (last `prev` steps)
        hist_mask = ~np.all(history_np[ego] == 0, axis=1)
        hist_to_plot = history_np[ego][max(0, H - prev):H]
        hist_to_plot = hist_to_plot[~np.all(hist_to_plot == 0, axis=1)]
        if hist_to_plot.shape[0] > 0:
            plt.plot(hist_to_plot[:, 0], hist_to_plot[:, 1], 'b.-', label='History')

        # Ground-truth future (first `next` steps)
        fut_to_plot = future_np[ego][:next]
        fut_to_plot = fut_to_plot[~np.all(fut_to_plot == 0, axis=1)]
        if fut_to_plot.shape[0] > 0:
            plt.plot(fut_to_plot[:, 0], fut_to_plot[:, 1], 'g.-', label='GT Future')

        # Predicted (first `next` of 12)
        pred_to_plot = preds_np[ego][:next]
        pred_to_plot = pred_to_plot[~np.all(pred_to_plot == 0, axis=1)]
        if pred_to_plot.shape[0] > 0:
            plt.plot(pred_to_plot[:, 0], pred_to_plot[:, 1], 'r.--', label='Predicted')

        # Ego last observed position
        if hist_to_plot.shape[0] > 0:
            ego_last = hist_to_plot[-1]
            plt.scatter([ego_last[0]], [ego_last[1]], marker='o', s=60, color='black', label='Ego (last)')

        # Neighbors (instantaneous positions at current frame)
        # neighbors_np[ego]: [N, 2]
                # Plot full trajectories for each neighbor (not just scatter)
        neighbor_positions = neighbors_np[ego]
        neighbor_count = 0
        for n_idx in range(neighbor_positions.shape[0]):
            if neighbor_count >= max_neighbors:
                break
            neigh_pos = neighbor_positions[n_idx]
            if np.all(neigh_pos == 0):
                continue
            # Find corresponding agent (by last observed position)
            agent_last = history_np[:, -1, :]
            dists = np.linalg.norm(agent_last - neigh_pos, axis=1)
            neighbor_agent_idx = np.argmin(dists)
            if neighbor_agent_idx == ego:
                continue  # skip ego
            color = cmap(neighbor_agent_idx)
            plt.plot(history_np[neighbor_agent_idx, :, 0], history_np[neighbor_agent_idx, :, 1], '--', color=color, alpha=0.5,
                     label=f'Neighbor {neighbor_count} History' if neighbor_count == 0 else None)
            plt.plot(future_np[neighbor_agent_idx, :, 0], future_np[neighbor_agent_idx, :, 1], '-', color=color, alpha=0.5,
                     label=f'Neighbor {neighbor_count} GT Future' if neighbor_count == 0 else None)
            plt.plot(preds_np[neighbor_agent_idx, :, 0], preds_np[neighbor_agent_idx, :, 1], ':', color=color, alpha=0.5,
                     label=f'Neighbor {neighbor_count} Predicted' if neighbor_count == 0 else None)
            neighbor_count += 1


        plt.title(f"Sample {sample_idx} | Agent {ego} (PFM non-learnable)")
        plt.xlabel("X"); plt.ylabel("Y")
        plt.axis('equal'); plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        plt.show()
    """
    Visualize observed, ground truth, neighbors, and predicted trajectories.
    """

    # ✅ Flexible unpacking
    sample = dataset[sample_idx]
    if len(sample) == 2:
        history, future = sample
        neighbors, goal = None, None
    elif len(sample) == 3:
        history, future, neighbors = sample
        goal = None
    elif len(sample) == 4:
        history, future, neighbors, goal = sample
    else:
        raise ValueError(f"Unexpected dataset output: {len(sample)} elements")

    # ✅ Move tensors to device
    history = history.unsqueeze(0).to(device)     # [B=1, A, H, 2]
    future = future.unsqueeze(0).to(device)       # [B=1, A, F, 2]
    if neighbors is not None:
        neighbors = neighbors.unsqueeze(0).to(device)  # [B=1, A, N, 2]
    if goal is not None:
        goal = goal.unsqueeze(0).to(device)            # [B=1, A, 2]

    # ✅ Model prediction
    with torch.no_grad():
        preds, _, _ = model(history, neighbors, goal)  # [B=1, A, T, 2]

    history = history[0].cpu()
    future = future[0].cpu()
    preds = preds[0].cpu()
    if neighbors is not None:
        neighbors = neighbors[0].cpu()
    if goal is not None:
        goal = goal[0].cpu()

    plt.figure(figsize=(8, 8))

    # Plot observed history
    for i in range(history.shape[0]):
        plt.plot(history[i, :, 0], history[i, :, 1], "bo-", label="History" if i == 0 else "")

    # Plot ground-truth future
    for i in range(future.shape[0]):
        plt.plot(future[i, :, 0], future[i, :, 1], "go--", label="Future (GT)" if i == 0 else "")

    # Plot predicted
    for i in range(preds.shape[0]):
        plt.plot(preds[i, :, 0], preds[i, :, 1], "ro-", label="Predicted" if i == 0 else "")

    # Plot neighbors if available
    if neighbors is not None:
        for i in range(neighbors.shape[0]):
            plt.scatter(neighbors[i, :, 0], neighbors[i, :, 1], c="gray", s=10, alpha=0.6,
                        label="Neighbors" if i == 0 else "")

    # Plot goal if available
    if goal is not None:
        for i in range(goal.shape[0]):
            plt.scatter(goal[i, 0], goal[i, 1], c="purple", marker="*", s=100,
                        label="Goal" if i == 0 else "")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Trajectory Visualization (Sample {sample_idx})")
    plt.legend()
    plt.axis("equal")
    plt.show()
    """
    Plot past, ground truth future, predicted future and neighbors for a given sample.
    
    Args:
        dataset: The dataset object containing trajectories and neighbor info
        model: The trained trajectory prediction model
        device: torch.device("cuda" or "cpu")
        sample_idx: Index of the sample in the dataset
        prev: Number of observed timesteps
        next: Number of timesteps to predict
    """
    model.eval()

    # Get data sample
    obs, fut, neighbors = dataset[sample_idx]
    # obs: (prev, 2)
    # fut: (next, 2)
    # neighbors: (num_neighbors, prev, 2)

    obs = obs.to(device).unsqueeze(0)  # (1, prev, 2)
    fut = fut.to(device).unsqueeze(0)  # (1, next, 2)
    neighbors = neighbors.to(device).unsqueeze(0)  # (1, num_neighbors, prev, 2)

    # Prediction
    with torch.no_grad():
        pred = model(obs, neighbors, pred_len=next)  # (1, next, 2)

    obs = obs.squeeze(0).cpu().numpy()
    fut = fut.squeeze(0).cpu().numpy()
    pred = pred.squeeze(0).cpu().numpy()
    neighbors = neighbors.squeeze(0).cpu().numpy()

    # Plot
    plt.figure(figsize=(8, 8))
    # Past trajectory
    plt.plot(obs[:, 0], obs[:, 1], "bo-", label="Observed (Past)")
    # Ground truth
    plt.plot(fut[:, 0], fut[:, 1], "go-", label="Ground Truth (Future)")
    # Predicted
    plt.plot(pred[:, 0], pred[:, 1], "ro--", label="Predicted (Future)")

    # Neighbors
    for n in range(min(12, neighbors.shape[0])):  # plot up to 12 neighbors
        plt.plot(neighbors[n, :, 0], neighbors[n, :, 1], "k--", alpha=0.5)

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Trajectory Prediction - Sample {sample_idx}")
    plt.grid(True)
    plt.show()





# =============================
# Example Dataset Loader (Your PFM_TrajectoryDataset class here)
# =============================
class PFM_TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, history_len=8, prediction_len=12):
        self.data = self.load_data(file_path)
        self.history_len = history_len
        self.prediction_len = prediction_len
        self.valid_frames = self._get_valid_frames()

    def load_data(self, file_path):
        data = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    frame, agent, x, y = map(float, parts)
                    frame, agent = int(frame), int(agent)
                    if frame not in data:
                        data[frame] = {}
                    data[frame][agent] = torch.tensor([x, y], dtype=torch.float32)
        return data

    def _get_valid_frames(self):
        all_frames = sorted(self.data.keys())
        valid_frames = []
        for frame in all_frames:
            history_start = frame - self.history_len + 1
            future_end = frame + self.prediction_len
            if history_start >= min(all_frames) and future_end <= max(all_frames):
                valid_frames.append(frame)
        return valid_frames

    def __len__(self):
        return len(self.valid_frames)

    def __getitem__(self, idx):
        frame = self.valid_frames[idx]
        if frame not in self.data:
            return (torch.zeros(0, self.history_len, 2),
                    torch.zeros(0, self.prediction_len, 2),
                    torch.zeros(0, 0, 2),
                    torch.zeros(0, 2))

        agents = list(self.data[frame].keys())
        num_agents = len(agents)
        if num_agents == 0:
            return (torch.zeros(0, self.history_len, 2),
                    torch.zeros(0, self.prediction_len, 2),
                    torch.zeros(0, 0, 2),
                    torch.zeros(0, 2))

        history = torch.zeros(num_agents, self.history_len, 2)
        future = torch.zeros(num_agents, self.prediction_len, 2)
        goals = torch.zeros(num_agents, 2)

        for i, agent in enumerate(agents):
            for t in range(self.history_len):
                hist_frame = frame - (self.history_len - 1 - t)
                if hist_frame in self.data and agent in self.data[hist_frame]:
                    history[i, t] = self.data[hist_frame][agent]
            for t in range(self.prediction_len):
                fut_frame = frame + t + 1
                if fut_frame in self.data and agent in self.data[fut_frame]:
                    future[i, t] = self.data[fut_frame][agent]
            non_zero_mask = torch.any(future[i] != 0, dim=1)
            if non_zero_mask.any():
                last_valid_idx = torch.where(non_zero_mask)[0][-1]
                goals[i] = future[i, last_valid_idx]
            else:
                goals[i] = self.data[frame][agent]

        neighbors_list = []
        for i, agent in enumerate(agents):
            agent_neighbors = []
            for other_agent in self.data[frame]:
                if other_agent != agent:
                    agent_neighbors.append(self.data[frame][other_agent])
            if agent_neighbors:
                neighbors_tensor = torch.stack(agent_neighbors)
            else:
                neighbors_tensor = torch.zeros(1, 2)
            neighbors_list.append(neighbors_tensor)

        if neighbors_list:
            max_neighbors = 12
            padded_neighbors = torch.zeros(num_agents, max_neighbors, 2)
            for i, neighbor_tensor in enumerate(neighbors_list):
                if neighbor_tensor.shape[0] > 0:
                    ego_pos = history[i, -1].unsqueeze(0)
                    dists = torch.norm(neighbor_tensor - ego_pos, dim=1)
                    sorted_idx = torch.argsort(dists)
                    sorted_neighbors = neighbor_tensor[sorted_idx]
                    top_neighbors = sorted_neighbors[:max_neighbors]
                    padded_neighbors[i, :top_neighbors.shape[0]] = top_neighbors
            neighbors = padded_neighbors
        else:
            neighbors = torch.zeros(num_agents, 1, 2)

        mask = torch.ones(history.shape[0], dtype=torch.bool)
        i = 0
        while i<history.shape[0]:
            for seq in range(history.shape[1]):
                if history[i,seq,0] == 0 and history[i,seq,1]==0:
                    mask[i] = False
            for seq in range(future.shape[1]):
                if future[i,seq,0] == 0 and future[i,seq,1]==0:
                    mask[i] = False
            i+=1
        history = history[mask]
        future = future[mask]
        neighbors = neighbors[mask]
        goals = goals[mask]

        return history, future, neighbors, goals

# =============================
# Usage Example
# =============================
# Load your model
# dataset must return (history, future, neighbors, goals)
dataset = PFM_TrajectoryDataset("/home/dasari-raj-vamsi/Desktop/iDataHub_running/data/combined_annotations.csv", history_len=8, prediction_len=12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.pfm_no_learnable import PFMOnlyModelNoLearnable
model = PFMOnlyModelNoLearnable().to(device)

plot_trajectories(dataset, model, device, sample_idx=0, prev=12, next=12)  # neighbors capped at 12 by default

