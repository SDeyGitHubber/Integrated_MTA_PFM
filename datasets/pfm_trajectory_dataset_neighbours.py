# import torch
# from torch.utils.data import Dataset


# class PFM_TrajectoryDataset_neighbours(Dataset):
#     def __init__(self, file_path, history_len=8, prediction_len=12, max_neighbors=12):
#         self.data = self.load_data(file_path)
#         self.history_len = history_len
#         self.prediction_len = prediction_len
#         self.max_neighbors = max_neighbors
#         self.valid_frames = self._get_valid_frames()

#     def load_data(self, file_path):
#         data = {}
#         with open(file_path, "r") as file:
#             for line in file:
#                 parts = line.strip().split(",")
#                 if len(parts) == 4:
#                     frame, agent, x, y = map(float, parts)
#                     frame, agent = int(frame), int(agent)
#                     if frame not in data:
#                         data[frame] = {}
#                     data[frame][agent] = torch.tensor([x, y], dtype=torch.float32)
#         return data

#     def _get_valid_frames(self):
#         all_frames = sorted(self.data.keys())
#         valid_frames = []
#         for frame in all_frames:
#             history_start = frame - self.history_len + 1
#             future_end = frame + self.prediction_len
#             if history_start >= min(all_frames) and future_end <= max(all_frames):
#                 valid_frames.append(frame)
#         return valid_frames

#     def __len__(self):
#         return len(self.valid_frames)

#     def __getitem__(self, idx):
#         frame = self.valid_frames[idx]
#         if frame not in self.data:
#             print("Dataset __getitem__: invalid frame, returning empty tensors")
#             return (torch.zeros(0, self.history_len, 2),
#                     torch.zeros(0, self.prediction_len, 2),
#                     torch.zeros(0, self.max_neighbors, self.history_len, 2),
#                     torch.zeros(0, 2))
    
#         agents = list(self.data[frame].keys())
#         num_agents = len(agents)
#         history = torch.zeros(num_agents, self.history_len, 2)
#         future = torch.zeros(num_agents, self.prediction_len, 2)
#         goals = torch.zeros(num_agents, 2)

#         for i, agent in enumerate(agents):
#             for t in range(self.history_len):
#                 hist_frame = frame - (self.history_len - 1 - t)
#                 if hist_frame in self.data and agent in self.data[hist_frame]:
#                     history[i, t] = self.data[hist_frame][agent]
#             for t in range(self.prediction_len):
#                 fut_frame = frame + t + 1
#                 if fut_frame in self.data and agent in self.data[fut_frame]:
#                     future[i, t] = self.data[fut_frame][agent]
#             non_zero_mask = torch.any(future[i] != 0, dim=1)
#             if non_zero_mask.any():
#                 last_valid_idx = torch.where(non_zero_mask)[0][-1]
#                 goals[i] = future[i, last_valid_idx]
#             else:
#                 goals[i] = self.data[frame][agent]

#         neighbor_histories = torch.zeros(num_agents, self.max_neighbors, self.history_len, 2)

#         for i, agent in enumerate(agents):
#             other_agents = [a for a in agents if a != agent][: self.max_neighbors]
#             for n_idx, neighbor in enumerate(other_agents):
#                 for t in range(self.history_len):
#                     hist_frame = frame - (self.history_len - 1 - t)
#                     if hist_frame in self.data and neighbor in self.data[hist_frame]:
#                         neighbor_histories[i, n_idx, t] = self.data[hist_frame][neighbor]

#         mask = torch.ones(history.shape[0], dtype=torch.bool)
#         for i in range(history.shape[0]):
#             if not torch.any(history[i]): mask[i] = False
#             if not torch.any(future[i]): mask[i] = False
    
#         history = history[mask]
#         future = future[mask]
#         neighbor_histories = neighbor_histories[mask]
#         goals = goals[mask]

#         print("[DATASET] __getitem__ output shapes:",
#               "history", history.shape,
#               "future", future.shape,
#               "neighbor_histories", neighbor_histories.shape,
#               "goals", goals.shape)

#         return history, future, neighbor_histories, goals
import torch
from torch.utils.data import Dataset

class PFM_TrajectoryDataset_neighbours(Dataset):
    def __init__(self, file_path, history_len=8, prediction_len=12, max_neighbors=12):
        self.data = self.load_data(file_path)
        self.history_len = history_len
        self.prediction_len = prediction_len
        self.max_neighbors = max_neighbors
        self.valid_frames = self._get_valid_frames()

    def load_data(self, file_path):
        data = {}
        print(f"[DL-LOAD] Loading data file: {file_path}")
        with open(file_path, "r") as file:
            for ln, line in enumerate(file, start=1):
                parts = line.strip().split(",")
                if len(parts) == 4:
                    frame, agent, x, y = map(float, parts)
                    frame, agent = int(frame), int(agent)
                    if frame not in data:
                        data[frame] = {}
                    data[frame][agent] = torch.tensor([x, y], dtype=torch.float32)
                else:
                    print(f"[DL-LOAD][WARN] Line {ln}: bad format: {line.strip()}")
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
            # Return empty tensors for all five outputs
            return (torch.zeros(0, self.max_neighbors + 1, self.history_len, 2),
                    torch.zeros(0, self.prediction_len, 2),
                    torch.zeros(0, self.max_neighbors, self.history_len, 2),
                    torch.zeros(0, 2),
                    torch.zeros(0, self.max_neighbors + 1, 2))

        agents = list(self.data[frame].keys())
        num_agents = len(agents)

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

        neighbor_histories = torch.zeros(num_agents, self.max_neighbors, self.history_len, 2)
        neighbor_goals = torch.zeros(num_agents, self.max_neighbors, 2)

        for i, agent in enumerate(agents):
            ego_pos = self.data[frame][agent]
            other_agents_with_dist = []
            for other_agent in agents:
                if other_agent == agent:
                    continue
                other_pos = self.data[frame][other_agent]
                dist = torch.norm(ego_pos - other_pos).item()
                other_agents_with_dist.append((other_agent, dist))
            other_agents_with_dist.sort(key=lambda x: x[1])
            other_agents = [x[0] for x in other_agents_with_dist[:self.max_neighbors]]

            for n_idx, neighbor in enumerate(other_agents):
                for t in range(self.history_len):
                    hist_frame = frame - (self.history_len - 1 - t)
                    if hist_frame in self.data and neighbor in self.data[hist_frame]:
                        neighbor_histories[i, n_idx, t] = self.data[hist_frame][neighbor]
                neighbor_future = torch.zeros(self.prediction_len, 2)
                for t in range(self.prediction_len):
                    fut_frame = frame + t + 1
                    if fut_frame in self.data and neighbor in self.data[fut_frame]:
                        neighbor_future[t] = self.data[fut_frame][neighbor]
                neighbor_non_zero_mask = torch.any(neighbor_future != 0, dim=1)
                if neighbor_non_zero_mask.any():
                    neighbor_last_valid_idx = torch.where(neighbor_non_zero_mask)[0][-1]
                    neighbor_goals[i, n_idx] = neighbor_future[neighbor_last_valid_idx]
                else:
                    if frame in self.data and neighbor in self.data[frame]:
                        neighbor_goals[i, n_idx] = self.data[frame][neighbor]
                    else:
                        neighbor_goals[i, n_idx] = torch.zeros(2)

        # Mask invalid agents
        mask = torch.ones(num_agents, dtype=torch.bool)
        for i in range(history.shape[0]):
            if not torch.any(history[i]) or not torch.any(future[i]) or torch.any(torch.all(neighbor_histories[i] == 0, dim=(1, 2))):
                mask[i] = False

        history = history[mask]
        future = future[mask]
        goals = goals[mask]
        neighbor_histories = neighbor_histories[mask]
        neighbor_goals = neighbor_goals[mask]

        # Concatenate ego history as first entity
        ego_history = history.unsqueeze(1)  # [num_agents, 1, H, D]
        history_neighbors = torch.cat((ego_history, neighbor_histories), dim=1)  # [num_agents, 1+max_neighbors, H, D]

        # Expand goals to match
        expanded_goals = torch.zeros(history_neighbors.shape[0], self.max_neighbors + 1, 2)
        for i in range(goals.shape[0]):
            expanded_goals[i, 0, :] = goals[i]
            for j in range(self.max_neighbors):
                if j < neighbor_goals.shape[1]:
                    expanded_goals[i, j + 1, :] = neighbor_goals[i, j]
                else:
                    expanded_goals[i, j + 1, :] = goals[i]

        print("[DATASET] __getitem__ output shapes:",
              "history_neighbors", history_neighbors.shape,
              "future", future.shape,
              "neighbor_histories", neighbor_histories.shape,
              "goals", goals.shape,
              "expanded_goals", expanded_goals.shape)

        return history_neighbors, future, neighbor_histories, goals, expanded_goals