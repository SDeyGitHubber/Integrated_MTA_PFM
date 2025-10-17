import torch
import numpy as np
import matplotlib.pyplot as plt

from models.mta_pfm_model import CheckpointedIntegratedMTAPFMModel
from datasets.pfm_trajectory_dataset import PFM_TrajectoryDataset


def plot_agents_individually_with_neighbors(dataset, device, sample_idx=0,
                                            prev=12, next=24,
                                            max_neighbors=12,
                                            model=None):
    """
    Plot individual agent trajectories (history, GT future, predicted) with neighbors.
    Compatible with CheckpointedIntegratedMTAPFMModel.

    Args:
        dataset: returns (history, future, neighbors, goals)
        device: 'cuda' or 'cpu'
        sample_idx: index of example in dataset to visualize
        prev: number of history steps to plot (<= history_len)
        next: number of future steps to plot (<= prediction length, usually 12)
        max_neighbors: max neighbors to plot per agent
        model: instantiated CheckpointedIntegratedMTAPFMModel with loaded weights
    """

    history, future, neighbors, goals = dataset[sample_idx]

    if history.shape[0] == 0:
        print(f"No agents in sample {sample_idx}")
        return

    history_np = history.cpu().numpy()
    future_np = future.cpu().numpy()

    with torch.no_grad():
        history_in = history.unsqueeze(0).to(device)    # Shape: [1, num_agents, hist_len, 2]
        neighbors_in = neighbors.unsqueeze(0).to(device)  # Shape: [1, num_agents, num_neighbors, 2]
        goals_in = goals.unsqueeze(0).to(device)         # Shape: [1, num_agents, 2]

        pred_out, coeff_mean, coeff_var = model(history_in, neighbors_in, goals_in)
        pred_np = pred_out.squeeze(0).cpu().numpy()      # Shape: [num_agents, 12, 2]

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

            if np.all(neigh_pos.cpu().numpy() == 0):
                continue

            # Find closest agent in history for coloring
            dists = np.linalg.norm(history_np[:, -1, :] - neigh_pos.cpu().numpy(), axis=1)
            closest_neighbor_idx = np.argmin(dists)

            if closest_neighbor_idx == ego_agent_idx:
                continue  # skip ego itself

            plot_traj(closest_neighbor_idx, f'(Neighbor {neighbor_count})', '-.', colors(closest_neighbor_idx))
            neighbor_count += 1

        plt.title(f"Agent {ego_agent_idx} and Neighbors (Integrated MTA+PFM Predictions)")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(fontsize='small', loc='best')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ---- Usage Example ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model and load your trained weights (adjust checkpoint path)
    model = CheckpointedIntegratedMTAPFMModel().to(device)
    checkpoint_path = "/home/dasari-raj-vamsi/Desktop/iDataHub_running/checkpoints/integrated_pfm_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load dataset
    dataset_path = '/home/dasari-raj-vamsi/Desktop/iDataHub_running/data/combined_annotations.csv'
    dataset = PFM_TrajectoryDataset(dataset_path, history_len=8, prediction_len=12)

    # Visualize the first sample (change sample_idx as desired)
    plot_agents_individually_with_neighbors(dataset, device, sample_idx=0, prev=8, next=12, model=model)
