#  ACTUAL ONE
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from models.mta_pfm_model_neighbours import CheckpointedIntegratedMTAPFM_neighbours
# from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
# from utils.collate_mta_pfm_neighbours import collate_fn


# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_agents_with_neighbors(dataset, model, device, sample_idx=0, max_neighbors=12):
#     """
#     Visualizes agent and neighbor trajectories, ground truths, and predictions, with detailed per-entity diagnostics and logs.

#     Args:
#         dataset (PFM_TrajectoryDataset_neighbours): Source dataset with neighbor histories and goals.
#         model (nn.Module): Trained trajectory forecasting model.
#         device (torch.device): CPU or CUDA device for inference.
#         sample_idx (int): Index of the sample to visualize.
#         max_neighbors (int): Max number of neighbors to display per agent.

#     Writes:
#         Results and debug stats to "plot_debug_output.txt", and plots to matplotlib figures.
#     """
#     frame_id = dataset.valid_frames[sample_idx]
#     print(f"\nVisualizing data for Frame ID: {frame_id}")

#     agents = list(dataset.data[frame_id].keys())
#     print(f"All agents in frame: {agents}")

#     neighbor_histories, goals, expanded_goals = dataset[sample_idx]

#     print("\n--- Raw Batch Shapes ---")
#     print("neighbor_histories:", neighbor_histories.shape)
#     print("goals:", goals.shape)
#     print("expanded_goals:", expanded_goals.shape)

#     if neighbor_histories.shape[0] == 0:
#         print(f"No agents present in sample {sample_idx}, Frame {frame_id}")
#         return

#     if neighbor_histories.dim() == 4:
#         neighbor_histories = neighbor_histories.unsqueeze(0)

#     # ego_history = torch.zeros(B, A, 1, H, D, device=neighbor_histories.device) #concat in the dataloader iself and not needed in here
#     # history_neighbors = torch.cat((ego_history, neighbor_histories), dim=2)

#     history_neighbors = neighbor_histories.to(device)
#     B, A, N, H, D = neighbor_histories.shape

#     expanded_goals = expanded_goals.unsqueeze(0).to(device) if expanded_goals.dim() == 3 else expanded_goals.to(device)

#     model.eval()
#     with torch.no_grad():
#         adjusted_preds, decoded_preds, coeff_mean, coeff_var = model(history_neighbors, expanded_goals)

#     print("\n--- Model Outputs ---")
#     print("adjusted_preds:", adjusted_preds.shape)
#     print("decoded_preds:", decoded_preds.shape)

#     history_neighbors_np = history_neighbors.squeeze(0).cpu().numpy()
#     decoded_np = decoded_preds.squeeze(0).cpu().numpy()
#     adjusted_np = adjusted_preds.squeeze(0).cpu().numpy()
#     expanded_goals_np = expanded_goals.squeeze(0).cpu().numpy()

#     # Write debug info to file
#     with open("plot_debug_output.txt", "a") as f:
#         num_to_show = min(12, history_neighbors_np.shape[0])
#         ent_to_show = min(13, decoded_np.shape[2])
#         for i in range(num_to_show):
#             f.write(f"\nEgo Agent Index {i}:\n")
#             f.write(f"    History: {history_neighbors_np[i, 0].tolist()}\n")
#             f.write(f"    Last position: {history_neighbors_np[i, 0, -1].tolist()}\n")
#             # no explicit future in this structure here
#             f.write(f"    Decoded Prediction: {decoded_np[i, 0].tolist()}\n")
#             f.write(f"    Adjusted Prediction: {adjusted_np[i, 0].tolist()}\n")
#             f.write(f"    Goal: {expanded_goals_np[i, 0].tolist()}\n")
#             f.write("    --- Neighbor Details ---\n")
#             for j in range(1, ent_to_show):
#                 f.write(f"      Neighbor Index {j}:\n")
#                 f.write(f"        History: {history_neighbors_np[i, j].tolist()}\n")
#                 f.write(f"        Last position: {history_neighbors_np[i, j, -1].tolist()}\n")
#                 f.write(f"        Decoded Prediction: {decoded_np[i, j].tolist()}\n")
#                 f.write(f"        Adjusted Prediction: {adjusted_np[i, j].tolist()}\n")
#                 f.write(f"        Goal: {expanded_goals_np[i, j].tolist()}\n")

#     colors = plt.cm.get_cmap('tab10')

#     for i, agent_idx in enumerate(range(min(5, history_neighbors_np.shape[0]))):
#         agent_id = agents[agent_idx] if agent_idx < len(agents) else f"Unknown_{agent_idx}"
#         print(f"\nPlotting Ego Agent idx: {agent_idx}, ID: {agent_id}")

#         base_color = colors(i)

#         def plot_traj(arr, style, label, color):
#             print(f"  plot_traj -> {label}, arr.shape={arr.shape}")
#             mask = ~np.all(arr == 0, axis=1)
#             plt.plot(arr[mask, 0], arr[mask, 1], style, label=label, color=color)

#         # Ego agent
#         plot_traj(history_neighbors_np[agent_idx, 0], 'o--', f'Agent {agent_id} History', 'blue')
#         plot_traj(decoded_np[agent_idx, 0], 'x-', f'Agent {agent_id} Decoded Prediction', 'purple')
#         plot_traj(adjusted_np[agent_idx, 0], '--', f'Agent {agent_id} Adjusted Prediction', 'green')

#         plt.scatter(expanded_goals_np[agent_idx, 0, 0], expanded_goals_np[agent_idx, 0, 1],
#                     s=150, marker='*', edgecolors='k', color=base_color, label=f'Agent {agent_id} Goal')

#         neighbors_plotted = 0
#         for nbr_idx in range(1, decoded_np.shape[2]):
#             if neighbors_plotted >= max_neighbors:
#                 break
#             if not decoded_np[agent_idx, nbr_idx].any():
#                 continue
#             if nbr_idx < len(agents):
#                 nbr_id = agents[nbr_idx]
#             else:
#                 nbr_id = f"Unknown_{nbr_idx}"

#             nbr_color = colors((i + nbr_idx) % 10)

#             plot_traj(history_neighbors_np[agent_idx, nbr_idx], 'o--', f'Neighbor {nbr_id} History', 'gray')
#             plot_traj(decoded_np[agent_idx, nbr_idx], 'x-', f'Neighbor {nbr_id} Decoded Prediction', 'purple')
#             plot_traj(adjusted_np[agent_idx, nbr_idx], '--', f'Neighbor {nbr_id} Adjusted Prediction', 'green')

#             plt.scatter(expanded_goals_np[agent_idx, nbr_idx, 0], expanded_goals_np[agent_idx, nbr_idx, 1],
#                         s=100, marker='*', edgecolors='k', alpha=0.6, color=nbr_color, label=f'Neighbor {nbr_id} Goal')

#             neighbors_plotted += 1

#         plt.title(f'Frame: {frame_id} - Agent {agent_id} and Neighbors')
#         plt.xlabel('X Position')
#         plt.ylabel('Y Position')
#         plt.legend(fontsize=8)
#         plt.grid(True)
#         plt.axis('equal')
#         plt.show()




# import torch
# # Example paths (update as needed)
# model_path = '/home/amarnath/Desktop/MTP_running/Integrated_MTA_PFM/checkpoints/mta_pfm_integrated_neighours_own_model.pth_epoch10.pth'
# data_path = '/home/amarnath/Desktop/MTP_running/Integrated_MTA_PFM/data/combined_annotations.csv'

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load dataset and model
# dataset = PFM_TrajectoryDataset_neighbours(data_path, history_len=8, prediction_len=12)
# model = CheckpointedIntegratedMTAPFM_neighbours()

# # Load checkpoint dict (do NOT call it 'cp' or 'checkpoint')
# ckpt = torch.load(model_path, map_location=device)
# if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
#     state_dict = ckpt["model_state_dict"]
# else:
#     print("[INFO] Loaded checkpoint without 'model_state_dict' key, assuming raw state dict.")
#     state_dict = ckpt
# model.load_state_dict(state_dict, strict=False)
# model.to(device)
# model.eval()
# print("[INFO] Model weights successfully loaded.")

# # Now you can run your visualization or inference
# plot_agents_with_neighbors(dataset, model, device, sample_idx=0, max_neighbors=12)

# # python ./testing_visualisation/mta_pfm_integrated_neighbours.py --checkpoint checkpoints/mta_pfm_trajectory_neighbours_model_own.pth 
# # --data_path data/combined_annotations.csv --cuda --sample_idx 0 --max_neighbors 12


import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import os

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)


def plot_agents_with_neighbors(dataset, model, device, sample_idx=0, max_neighbors=12):
    frame_id = dataset.valid_frames[sample_idx]
    print(f"\nVisualizing data for Frame ID: {frame_id}")

    agents = list(dataset.data[frame_id].keys())
    print(f"All agents in frame: {agents}")

    history_neighbors, future, neighbor_histories, goals, expanded_goals = dataset[sample_idx]

    print("\n--- Raw Batch Shapes ---")
    print("history_neighbors:", history_neighbors.shape)
    print("future:", future.shape)
    print("neighbor_histories:", neighbor_histories.shape)
    print("goals:", goals.shape)
    print("expanded_goals:", expanded_goals.shape)

    if history_neighbors.shape[0] == 0:
        print(f"No agents present in sample {sample_idx}, Frame {frame_id}")
        return

    if history_neighbors.dim() == 4:
        history_neighbors = history_neighbors.unsqueeze(0)  # [1, agents, ent, H, D]

    if expanded_goals.dim() == 3:
        expanded_goals = expanded_goals.unsqueeze(0)

    history_neighbors = history_neighbors.to(device)
    expanded_goals = expanded_goals.to(device)

    model.eval()
    with torch.no_grad():
        adjusted_preds, decoded_preds, coeff_mean, coeff_var = model(history_neighbors, expanded_goals)

    print("\n--- Model Outputs ---")
    print("adjusted_preds:", adjusted_preds.shape)
    print("decoded_preds:", decoded_preds.shape)

    history_neighbors_np = history_neighbors.squeeze(0).cpu().numpy()
    future_np = future.cpu().numpy()  # Ensure future is on CPU and numpy
    decoded_np = decoded_preds.squeeze(0).cpu().numpy()
    adjusted_np = adjusted_preds.squeeze(0).cpu().numpy()
    expanded_goals_np = expanded_goals.squeeze(0).cpu().numpy()

    with open("plot_debug_output.txt", "a") as f:
        num_to_show = min(12, history_neighbors_np.shape[0])
        ent_to_show = min(13, decoded_np.shape[2])
        for i in range(num_to_show):
            f.write(f"\nEgo Agent Index {i}:\n")
            f.write(f"    History: {history_neighbors_np[i, 0].tolist()}\n")
            f.write(f"    Last position: {history_neighbors_np[i, 0, -1].tolist()}\n")
            f.write(f"    Decoded Prediction: {decoded_np[i, 0].tolist()}\n")
            f.write(f"    Adjusted Prediction: {adjusted_np[i, 0].tolist()}\n")
            f.write(f"    Goal: {expanded_goals_np[i, 0].tolist()}\n")
            f.write(f"    Future (Ground Truth): {future_np[i].tolist()}\n")
            f.write("    --- Neighbor Details ---\n")
            for j in range(1, ent_to_show):
                f.write(f"      Neighbor Index {j}:\n")
                f.write(f"        History: {history_neighbors_np[i, j].tolist()}\n")
                f.write(f"        Last position: {history_neighbors_np[i, j, -1].tolist()}\n")
                f.write(f"        Decoded Prediction: {decoded_np[i, j].tolist()}\n")
                f.write(f"        Adjusted Prediction: {adjusted_np[i, j].tolist()}\n")
                f.write(f"        Goal: {expanded_goals_np[i, j].tolist()}\n")
                # Log neighbor's ground truth future if available and shape matches
                if j < future_np.shape[1]:
                    f.write(f"        Future (Ground Truth): {future_np[i, j].tolist()}\n")
                else:
                    f.write(f"        Future (Ground Truth): N/A\n")

    colors = plt.cm.get_cmap('tab10')

    for i, agent_idx in enumerate(range(min(5, history_neighbors_np.shape[0]))):
        agent_id = agents[agent_idx] if agent_idx < len(agents) else f"Unknown_{agent_idx}"
        print(f"\nPlotting Ego Agent idx: {agent_idx}, ID: {agent_id}")

        plt.figure(figsize=(12, 12))

        base_color = colors(i)

        def plot_traj(arr, style, label, color):
            print(f"  plot_traj -> {label}, arr.shape={arr.shape}")
            mask = ~np.all(arr == 0, axis=1)
            plt.plot(arr[mask, 0], arr[mask, 1], style, label=label, color=color)

        plot_traj(history_neighbors_np[agent_idx, 0], 'o--', f'Agent {agent_id} History', 'blue')
        plot_traj(decoded_np[agent_idx, 0], 'x-', f'Agent {agent_id} Decoded Prediction', 'purple')
        plot_traj(adjusted_np[agent_idx, 0], '--', f'Agent {agent_id} Adjusted Prediction', 'green')

        plt.scatter(expanded_goals_np[agent_idx, 0, 0], expanded_goals_np[agent_idx, 0, 1],
                    s=150, marker='*', edgecolors='k', color=base_color, label=f'Agent {agent_id} Goal')

        neighbors_plotted = 0
        for nbr_idx in range(1, decoded_np.shape[2]):
            if neighbors_plotted >= max_neighbors:
                break
            if not decoded_np[agent_idx, nbr_idx].any():
                continue
            nbr_id = agents[nbr_idx] if nbr_idx < len(agents) else f"Unknown_{nbr_idx}"
            nbr_color = colors((i + nbr_idx) % 10)
            plot_traj(history_neighbors_np[agent_idx, nbr_idx], 'o--', f'Neighbor {nbr_id} History', 'gray')
            plot_traj(decoded_np[agent_idx, nbr_idx], 'x-', f'Neighbor {nbr_id} Decoded Prediction', 'purple')
            plot_traj(adjusted_np[agent_idx, nbr_idx], '--', f'Neighbor {nbr_id} Adjusted Prediction', 'green')
            plt.scatter(expanded_goals_np[agent_idx, nbr_idx, 0], expanded_goals_np[agent_idx, nbr_idx, 1],
                        s=100, marker='*', edgecolors='k', alpha=0.6, color=nbr_color, label=f'Neighbor {nbr_id} Goal')
            neighbors_plotted += 1

        plt.title(f'Frame: {frame_id} - Agent {agent_id} and Neighbors')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.axis('equal')

        output_path = f"plots/frame_{frame_id}_agent_{agent_id}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_path}")
        plt.close()



# --- Main execution part ---
# Note: Assuming these imports are relative to your project structure
from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
from models.mta_pfm_model_neighbours import CheckpointedIntegratedMTAPFM_neighbours

model_path = '/home/amarnath/Desktop/MTP_running/Integrated_MTA_PFM/checkpoints/mta_pfm_integrated_neighours_own_model.pth_epoch10.pth'
data_path = '/home/amarnath/Desktop/MTP_running/Integrated_MTA_PFM/data/combined_annotations.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PFM_TrajectoryDataset_neighbours(data_path, history_len=8, prediction_len=12)
model = CheckpointedIntegratedMTAPFM_neighbours()

ckpt = torch.load(model_path, map_location=device)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
else:
    print("[INFO] Loaded checkpoint without 'model_state_dict' key, assuming raw state dict.")
    state_dict = ckpt
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print("[INFO] Model weights successfully loaded.")

plot_agents_with_neighbors(dataset, model, device, sample_idx=0, max_neighbors=12)

print("[INFO] Visualization script finished.")
# python3 -m testing_visualisation.mta_pfm_integrated_neighbours --checkpoint checkpoints/mta_pfm_integrated_neighours_own_model.pth_epoch10.pth --data_path data/combined_annotations.csv --cuda --sample_idx 0 --max_neighbors 12
