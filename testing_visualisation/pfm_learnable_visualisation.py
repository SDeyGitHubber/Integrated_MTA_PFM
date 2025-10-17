import torch
import numpy as np
import matplotlib.pyplot as plt
from models.pfm_model import PFMOnlyModel


def plot_agents_individually_with_neighbors(dataset, device, model, sample_idx=0,
                                           prev=12, next=24,
                                           max_neighbors=12):
    history, future, neighbors, goals = dataset[sample_idx]

    if history.shape[0] == 0:
        print(f"No agents in sample {sample_idx}")
        return

    history_np = history.numpy()
    future_np = future.numpy()

    # Use model for prediction
    with torch.no_grad():
        history_input = history.unsqueeze(0).to(device)    # shape: (1, num_agents, hist_len, 2)
        neighbors_input = neighbors.unsqueeze(0).to(device) # shape: (1, num_agents, num_neighbors, 2)
        goals_input = goals.unsqueeze(0).to(device)        # shape: (1, num_agents, 2)

        pred_out, _, _ = model(history_input, neighbors_input, goals_input)  # output: (1, num_agents, 12, 2)
        preds = pred_out.squeeze(0).cpu().numpy()  # (num_agents, 12, 2)

    num_agents = history_np.shape[0]
    colors = plt.cm.get_cmap('tab10', num_agents)

    for ego_idx in range(num_agents):
        plt.figure(figsize=(10, 8))

        def plot_traj(agent_idx, label_suffix, linestyle, color):
            traj_hist = history_np[agent_idx]
            traj_fut = future_np[agent_idx]
            traj_pred = preds[agent_idx]

            # Masks to filter out zero padded points
            mask_hist = ~np.all(traj_hist == 0, axis=1)
            mask_fut = ~np.all(traj_fut == 0, axis=1)
            mask_pred = ~np.all(traj_pred == 0, axis=1)

            plt.plot(traj_hist[mask_hist, 0], traj_hist[mask_hist, 1], linestyle,
                     color=color, label=f'Agent {agent_idx} History {label_suffix}')
            plt.plot(traj_fut[mask_fut, 0], traj_fut[mask_fut, 1], '-',
                     color=color, alpha=0.8, label=f'Agent {agent_idx} GT Future {label_suffix}')
            plt.plot(traj_pred[mask_pred, 0], traj_pred[mask_pred, 1], ':',
                     color=color, alpha=0.7, label=f'Agent {agent_idx} Predicted {label_suffix}')

        # Plot ego agent trajectory
        plot_traj(ego_idx, '(Ego)', '--', colors(ego_idx))

        # Plot neighbors
        neighbor_positions = neighbors[ego_idx]
        neighbor_count = 0

        for n_idx in range(neighbor_positions.shape[0]):
            if neighbor_count >= max_neighbors:
                break
            neigh_pos = neighbor_positions[n_idx]
            if torch.all(neigh_pos == 0):
                continue

            dists = torch.norm(history[:, -1, :] - neigh_pos, dim=1).cpu().numpy()
            closest_neighbor_idx = np.argmin(dists)

            if closest_neighbor_idx == ego_idx:
                continue  # skip the ego agent itself

            plot_traj(closest_neighbor_idx, f'(Neighbor {neighbor_count})', '-.', colors(closest_neighbor_idx))
            neighbor_count += 1

        plt.title(f"Agent {ego_idx} and Neighbors (PFM Predictions)")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(fontsize='small', loc='best')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


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

            # Set goal as last valid future point or current position
            nonzero_mask = torch.any(future[i] != 0, dim=1)
            if nonzero_mask.any():
                last_valid_idx = torch.where(nonzero_mask)[0][-1]
                goals[i] = future[i, last_valid_idx]
            else:
                goals[i] = self.data[frame][agent]

        neighbors_list = []
        for i, agent in enumerate(agents):
            neighbors_i = []
            for other_agent in self.data[frame]:
                if other_agent != agent:
                    neighbors_i.append(self.data[frame][other_agent])
            if neighbors_i:
                tensor_neighbors = torch.stack(neighbors_i)
            else:
                tensor_neighbors = torch.zeros(1, 2)
            neighbors_list.append(tensor_neighbors)

        if neighbors_list:
            max_neighbors = 12
            padded_neighbors = torch.zeros(num_agents, max_neighbors, 2)
            for i, neighbors_tensor in enumerate(neighbors_list):
                n = neighbors_tensor.shape[0]
                padded_neighbors[i, :n] = neighbors_tensor[:max_neighbors]
            neighbors = padded_neighbors
        else:
            neighbors = torch.zeros(num_agents, 1, 2)

        # Mask agents with no points in history/future
        mask = torch.ones(num_agents, dtype=torch.bool)
        for i in range(num_agents):
            if (history[i] == 0).all() or (future[i] == 0).all():
                mask[i] = False

        history = history[mask]
        future = future[mask]
        neighbors = neighbors[mask]
        goals = goals[mask]

        return history, future, neighbors, goals


# Usage example
if __name__ == "__main__":
    dataset = PFM_TrajectoryDataset('/home/dasari-raj-vamsi/Desktop/iDataHub_running/data/combined_annotations.csv', history_len=8, prediction_len=12)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import your model and load checkpoint appropriately

    model = PFMOnlyModel().to(device)
    checkpoint = torch.load('/home/dasari-raj-vamsi/Desktop/iDataHub_running/checkpoints/pfm_only_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    plot_agents_individually_with_neighbors(dataset, device, model, sample_idx=0)