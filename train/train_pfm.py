import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.pfm_trajectory_dataset import PFM_TrajectoryDataset
from models.pfm_model import PFMOnlyModel
from utils.collate import collate_fn
from torch.utils.data import DataLoader, random_split



import gc
import torch
import torch.nn as nn

def train_pfm_model(
    data_path,
    model_save_path,
    model_class,       # now PFMOnlyModel
    dataset_class,
    collate_fn,
    batch_size=32,
    epochs=3,
    learning_rate=1e-3,
    weight_decay=0.0,
    device=None
):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = dataset_class(data_path)
    loader =torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Monitor initial coefficients
    init = model.pfm.coeff_embedding.weight.data.clone()
    print("Initial coeffs sample:", init[:3])

    for epoch in range(epochs):
        model.train()
        for batch_idx, (hist, fut, neigh, goal) in enumerate(loader):
            hist, fut, neigh = hist.to(device), fut.to(device), neigh.to(device)
            goal = fut[:,:,-1,:].to(device)

            optimizer.zero_grad()
            pred, _, _ = model(hist, neigh, goal)
            loss = criterion(pred, fut)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                grads = model.pfm.coeff_embedding.weight.grad
                print(f"[E{epoch+1} B{batch_idx}] Loss={loss:.4f}, GradNorm={grads.norm():.6f}")
                print(" Coeff sample:", model.pfm.coeff_embedding.weight.data[:3])

    torch.save({'model_state_dict': model.state_dict()}, model_save_path)
    print("Model saved to", model_save_path)


if __name__ == "__main__":
    train_pfm_model(
        data_path="data/combined_annotations.csv",
        model_save_path="/checkpoints/pfm_nolearnable_trajectory_model_own_exp.pth",
        model_class=PFMOnlyModel,
        dataset_class=PFM_TrajectoryDataset,
        collate_fn=collate_fn,
        batch_size=8,
        epochs=1
    )