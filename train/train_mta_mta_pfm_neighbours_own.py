import torch
from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
from utils.collate_mta_pfm_neighbours import collate_fn

from models.mta_pfm_model_neighbours import CheckpointedIntegratedMTAPFM_neighbours  # MTA+PFM hybrid
from models.mta_model_neighbours import CheckpointedIntegratedMTAModelNeighbours       # MTA without PFM

from train.train_mta_pfm_neighbours import train_mta_pfm_model  # training func for hybrid
from train.train_mta_neighbours import train_mta_model     # training func for MTA-only


def main():
    print("Starting training script for MTA and MTA+PFM models")

    data_path = "data/combined_annotations.csv"
    batch_size = 1
    epochs = 10
    learning_rate = 0.0001
    patience = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nTraining MTA-only Model...")
    train_mta_model(
        data_path=data_path,
        model_save_path="checkpoints/mta_model_neighbours_own_model.pth",
        model_class=CheckpointedIntegratedMTAModelNeighbours,
        dataset_class=PFM_TrajectoryDataset_neighbours,
        collate_fn=collate_fn,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        device=device,
    )

    print("\nTraining MTA+PFM Integrated Model...")
    train_mta_pfm_model(
        data_path=data_path,
        model_save_path="checkpoints/mta_pfm_integrated_neighours_own_model.pth",
        model_class=CheckpointedIntegratedMTAPFM_neighbours,
        dataset_class=PFM_TrajectoryDataset_neighbours,
        collate_fn=collate_fn,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        device=device,
    )

    print("\nAll models trained successfully!")


if __name__ == "__main__":
    main()