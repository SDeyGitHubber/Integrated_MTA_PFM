import torch
import torch.nn as nn


class PotentialField(nn.Module):
    def __init__(self, goal, num_agents=1000, k_init=1.0, repulsion_radius=0.5):
        super().__init__()
        self.register_buffer('goal', torch.tensor(goal, dtype=torch.float32))
        self.repulsion_radius = repulsion_radius
        self.coeff_embedding = nn.Embedding(num_agents, 3)
        self.coeff_embedding.weight.data.fill_(k_init)


    def forward(self, pos, predicted, neighbors, goal, coeffs):
        k1, k2, kr = coeffs[..., 0:1], coeffs[..., 1:2], coeffs[..., 2:3]
        if predicted.dim() == 3:
            Fp = k2 * (predicted - pos)
        else:
            # e.g., shape [B, A, 1, 2]
            Fp = k2 * (predicted[:, :, 0, :] - pos)
        
        Fg = k1 * (goal - pos)
        diffs = pos.unsqueeze(2) - neighbors
        dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
        mask = (dists < self.repulsion_radius).float()
        Fr = (kr.unsqueeze(2) * diffs / dists.pow(2) * mask).sum(dim=2)
        return Fg + Fp + Fr, coeffs


# === New PFMOnlyModel (replaces LSTM-based model) ===
class PFMOnlyModel(nn.Module):
    def __init__(self, goal=(4.2, 4.2), target_avg_speed=4.087,
                 speed_tolerance=0.15, num_agents=1000, dt=0.1):
        super().__init__()
        self.pfm = PotentialField(goal, num_agents)
        if target_avg_speed is None:
            raise ValueError("target_avg_speed required")
        self.min_speed = target_avg_speed * (1 - speed_tolerance)
        self.max_speed = target_avg_speed * (1 + speed_tolerance)
        self.dt = dt


    def apply_speed_constraints(self, preds, last_pos):
        B, A, T, _ = preds.shape
        out = preds.clone()
        cur = last_pos.clone()
        for t in range(T):
            disp = out[:, :, t] - cur
            sp = torch.norm(disp, dim=-1, keepdim=True)
            nz = sp > 0
            clipped = torch.clamp(sp, self.min_speed, self.max_speed)
            sp_final = torch.where(nz, clipped, sp)
            dir = disp / (sp + 1e-8)
            out[:, :, t] = cur + dir * sp_final
            cur = out[:, :, t].clone()
        return out


    def forward(self, history, neighbors, goal):
        B, A, H, _ = history.shape
        agent_ids = torch.arange(A).repeat(B, 1).to(history.device)
        coeffs = self.pfm.coeff_embedding(agent_ids)


        preds = torch.zeros(B, A, 12, 2, device=history.device)
        cur = history[:, :, -1, :].clone()
        cur_neighbors = neighbors.clone()
        coeff_list = []


        for t in range(12):
            if t == 0 and H >= 2:
                vel = history[:, :, -1, :] - history[:, :, -2, :]
                pred_slice = (cur + vel).unsqueeze(2)
            elif t == 0:
                pred_slice = cur.unsqueeze(2)
            else:
                pred_slice = preds[:, :, t-1:t, :].clone()


            forces_ego, cstep_ego = self.pfm(cur, pred_slice, cur_neighbors, goal, coeffs)
            nextp_ego = cur + forces_ego * self.dt
            preds[:, :, t, :] = nextp_ego
            cur = nextp_ego.detach()  # detach to save memory


            B_, A_, N, _ = cur_neighbors.shape
            cur_neighbors_flat = cur_neighbors.view(B_ * A_ * N, 2)
            pred_slice_neighbors = cur_neighbors_flat.unsqueeze(1)
            coeffs_neighbors = coeffs.unsqueeze(2).repeat(1, 1, N, 1).view(B_ * A_ * N, 3)


            # Represent neighbors of neighbors as an empty tensor to avoid memory load
            neighbors_neighbors = torch.empty(0, 0, 2, device=cur_neighbors.device)


            # Correctly expand goal by adding neighbors dimension (dim=2)
            goal_expanded = goal.unsqueeze(2).expand(B_, A_, N, 2).contiguous().view(B_ * A_ * N, 2)


            forces_neighbors, _ = self.pfm(
                pos=cur_neighbors_flat,
                predicted=pred_slice_neighbors,
                neighbors=neighbors_neighbors,  # empty tensor here
                goal=goal_expanded,
                coeffs=coeffs_neighbors
            )


            nextp_neighbors = cur_neighbors_flat + forces_neighbors * self.dt
            cur_neighbors = nextp_neighbors.view(B_, A_, N, 2).clone().detach()  # detach for memory efficiency


            coeff_list.append(cstep_ego)


        preds = self.apply_speed_constraints(preds, history[:, :, -1, :])
        stack = torch.stack(coeff_list, dim=0)
        return preds, stack.mean(), stack.var(unbiased=False)
    



    INITIAL CODE
    class PotentialField(nn.Module):
    def __init__(self, goal, num_agents=1000, k_init=1.0, repulsion_radius=0.5):
        super().__init__()
        self.register_buffer('goal', torch.tensor(goal, dtype=torch.float32))
        self.repulsion_radius = repulsion_radius
        self.coeff_embedding = nn.Embedding(num_agents, 3)
        self.coeff_embedding.weight.data.fill_(k_init)
    def forward(self, pos, predicted, neighbors, goal, coeffs):
        k1, k2, kr = coeffs[...,0:1], coeffs[...,1:2], coeffs[...,2:3]
        Fg = k1 * (goal - pos)
        Fp = k2 * (predicted[:,:,0,:] - pos)
        diffs = pos.unsqueeze(2) - neighbors
        dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
        mask = (dists < self.repulsion_radius).float()
        Fr = (kr.unsqueeze(2) * diffs / dists.pow(2) * mask).sum(dim=2)
        return Fg + Fp + Fr, coeffs
# === New PFMOnlyModel (replaces LSTM-based model) ===
class PFMOnlyModel(nn.Module):
    def __init__(self, goal=(4.2,4.2), target_avg_speed=4.087,
                 speed_tolerance=0.15, num_agents=1000, dt=0.1):
        super().__init__()
        self.pfm = PotentialField(goal, num_agents)
        if target_avg_speed is None:
            raise ValueError("target_avg_speed required")
        self.min_speed = target_avg_speed * (1 - speed_tolerance)
        self.max_speed = target_avg_speed * (1 + speed_tolerance)
        self.dt = dt
    def apply_speed_constraints(self, preds, last_pos):
        B,A,T,_ = preds.shape
        out = preds.clone()
        cur = last_pos.clone()
        for t in range(T):
            disp = out[:,:,t] - cur
            sp = torch.norm(disp, dim=-1, keepdim=True)
            nz = sp>0
            clipped = torch.clamp(sp, self.min_speed, self.max_speed)
            sp_final = torch.where(nz, clipped, sp)
            dir = disp/(sp+1e-8)
            out[:,:,t] = cur + dir*sp_final
            cur = out[:,:,t].clone()
        # print(out{:,:,-2},out{:,:,-1},torch.norm(out{:,:,-2} - out{:,:,-1}, dim=-1, keepdim=True))    #RKL Add
        return out
    def forward(self, history, neighbors, goal):
        B,A,H,_ = history.shape
        agent_ids = torch.arange(A).repeat(B,1).to(history.device)
        coeffs = self.pfm.coeff_embedding(agent_ids) #RKL1 cgange agent ids to history
        preds = torch.zeros(B,A,12,2,device=history.device)
        cur = history[:,:,-1,:].clone()
        coeff_list=[]
        for t in range(12):
            if t==0 and H>=2:
                vel = history[:,:,-1,:] - history[:,:,-2,:]
                pred_slice = (cur+vel).unsqueeze(2)
            elif t==0:
                pred_slice = cur.unsqueeze(2)
            else:
                pred_slice = preds[:,:,t-1:t,:].clone()
            forces, cstep = self.pfm(cur, pred_slice, neighbors, goal, coeffs) #RKL1 the current position i spred_slice, the future prediction should come from the neural network
            # RKL1 move neighbours by 1 step using potential field only
            nextp = cur + forces*self.dt
            preds[:,:,t] = nextp
            cur = nextp.clone()
            coeff_list.append(cstep)
        preds = self.apply_speed_constraints(preds, history[:,:,-1,:])
        stack = torch.stack(coeff_list,dim=0)
        return preds, stack.mean(), stack.var(unbiased=False) 