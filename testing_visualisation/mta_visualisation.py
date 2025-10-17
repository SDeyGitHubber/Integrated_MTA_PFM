import torch
import numpy as np
import matplotlib.pyplot as plt
from models.deep_lstm_model import DeepLSTMModel
from datasets.pfm_trajectory_dataset import PFM_TrajectoryDataset

# =============================
# 2. Replace pred_function usage
# =============================
import matplotlib.pyplot as plt
import numpy as np

import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_agents_individually_with_neighbors(dataset, device, sample_idx=0,
                                            prev=12, next=24,
                                            max_neighbors=12,
                                            model=None):  # pass your DeepLSTMModel instance here
    history, future, neighbors, goals = dataset[sample_idx]
    if history.shape[0] == 0:
        print(f"No agents in sample {sample_idx}")
        return
    history_np = history.numpy()
    future_np = future.numpy()
    with torch.no_grad():
        history_in = history.unsqueeze(0).to(device)     # [1, num_agents, hist_len, 2]
        neighbors_in = neighbors.unsqueeze(0).to(device)
        goals_in = goals.unsqueeze(0).to(device)
        pred_out, _, _ = model(history_in, neighbors_in, goals_in)  # Shape [1, num_agents, 12, 2]
        pred_np = pred_out.squeeze(0).cpu().numpy()
    num_agents = history_np.shape[0]
    colors = plt.cm.get_cmap('tab10', num_agents)
    for ego_agent_idx in range(num_agents):
        plt.figure(figsize=(10, 8))
        def plot_traj(agent_idx, label_suffix, linestyle, color):
            traj_hist = history_np[agent_idx]
            traj_fut = future_np[agent_idx]
            traj_pred = pred_np[agent_idx]
            mask_hist = ~np.all(traj_hist == 0, axis=1)
            mask_fut = ~np.all(traj_fut == 0, axis=1)
            mask_pred = ~np.all(traj_pred == 0, axis=1)
            plt.plot(traj_hist[mask_hist, 0], traj_hist[mask_hist, 1], linestyle,
                     color=color, label=f'Agent {agent_idx} History {label_suffix}')
            plt.plot(traj_fut[mask_fut, 0], traj_fut[mask_fut, 1], '-',
                     color=color, alpha=0.8, label=f'Agent {agent_idx} GT Future {label_suffix}')
            plt.plot(traj_pred[mask_pred, 0], traj_pred[mask_pred, 1], ':',
                     color=color, alpha=0.7, label=f'Agent {agent_idx} Predicted {label_suffix}')
        # Plot ego agent
        plot_traj(ego_agent_idx, '(Ego)', '--', colors(ego_agent_idx))
        # Plot neighbors (up to max_neighbors)
        neighbor_positions = neighbors[ego_agent_idx]
        neighbor_count = 0
        for n_idx in range(neighbor_positions.shape[0]):
            if neighbor_count >= max_neighbors:
                break
            neigh_pos = neighbor_positions[n_idx]
            if np.all(neigh_pos.numpy() == 0):
                continue
            dists = np.linalg.norm(history_np[:, -1, :] - neigh_pos.numpy(), axis=1)
            closest_neighbor_idx = np.argmin(dists)
            if closest_neighbor_idx == ego_agent_idx:
                continue  # skip ego itself
            plot_traj(closest_neighbor_idx, f'(Neighbor {neighbor_count})', '-.', colors(closest_neighbor_idx))
            neighbor_count += 1
        plt.title(f"Agent {ego_agent_idx} and Neighbors (DeepLSTM Predictions)")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(fontsize='small', loc='best')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# =============================
# Load the trained DeepLSTM model checkpoint
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model (must match constructor/parameters used during training)
model_mta = DeepLSTMModel().to(device)
checkpoint_path = "/home/dasari-raj-vamsi/Desktop/iDataHub_running/checkpoints/mta_trajectory_model_Zara_5eps.pth"
state = torch.load(checkpoint_path, map_location=device)
model_mta.load_state_dict(state['model_state_dict'])
model_mta.eval()

# =============================
# Load the dataset
# =============================
dataset = PFM_TrajectoryDataset('/home/dasari-raj-vamsi/Desktop/iDataHub_running/data/combined_annotations.csv', history_len=8, prediction_len=12)

# =============================
# Run the plotting function
# =============================
plot_agents_individually_with_neighbors(dataset, device, sample_idx=0, prev=8, next=12, model=model_mta)





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
# Load the trained DeepLSTM model checkpoint
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model (must match constructor/parameters used during training)
model_mta = DeepLSTMModel().to(device)
checkpoint_path = "/home/dasari-raj-vamsi/Desktop/iDataHub_running/checkpoints/mta_trajectory_model_Zara_5eps.pth"
state = torch.load(checkpoint_path, map_location=device)
model_mta.load_state_dict(state['model_state_dict'])
model_mta.eval()

# =============================
# Load the dataset
# =============================
dataset = PFM_TrajectoryDataset('data/combined_annotations.csv', history_len=8, prediction_len=12)

# =============================
# Run the plotting function
# =============================
plot_agents_individually_with_neighbors(dataset, device, sample_idx=0, prev=8, next=12, model=model_mta)

