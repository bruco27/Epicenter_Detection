import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from timm import create_model
from torchvision import transforms
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Define the PatchDataset class
class PatchDataset(Dataset):
    def __init__(self, h5_file_path, csv_file_path, transform=None):
        self.metadata_df = pd.read_csv(csv_file_path)
        self.h5_file_path = h5_file_path
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df) // 3

    def __getitem__(self, idx):
        start_idx = idx * 3
        end_idx = start_idx + 3
        rows = self.metadata_df.iloc[start_idx:end_idx]
        if len(rows) != 3:
            raise ValueError("Each image must have exactly 3 rows (one for each channel).")

        row = rows.iloc[0]
        interferogram = row['interferogram']
        patch_name = row['patches']

        coh_path = f"{interferogram}/coh/{patch_name}"
        i_path = f"{interferogram}/i/{patch_name}"
        q_path = f"{interferogram}/q/{patch_name}"

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            try:
                coh_data = np.array(h5_file[coh_path])
                i_data = np.array(h5_file[i_path])
                q_data = np.array(h5_file[q_path])
            except KeyError:
                raise ValueError(f"Path {coh_path}, {i_path}, or {q_path} not found in the HDF5 file")

            patch_data = np.stack([coh_data, i_data, q_data], axis=0)

        # Normalize center_lat and center_lon to [-1, 1]
        norm_center_lat = 2 * (row['center_lat'] + 90) / 180 - 1
        norm_center_lon = 2 * (row['center_lon'] + 180) / 360 - 1
        center_lat_channel = np.full_like(coh_data, norm_center_lat, dtype=np.float32)
        center_lon_channel = np.full_like(coh_data, norm_center_lon, dtype=np.float32)

        patch_data = np.stack([coh_data, i_data, q_data, center_lat_channel, center_lon_channel], axis=0)

        # Clamp extreme values
        patch_data = np.clip(patch_data, a_min=-1e3, a_max=1e3)

        # Convert to tensor and apply transformations
        patch_data = torch.tensor(patch_data, dtype=torch.float32)
        if self.transform:
            patch_data = self.transform(patch_data)

        # Normalize epicenter_lat and epicenter_lon to [-1, 1]
        norm_lat = 2 * (row['epicenter_lat'] + 90) / 180 - 1
        norm_lon = 2 * (row['epicenter_lon'] + 180) / 360 - 1
        target = torch.tensor([norm_lat, norm_lon], dtype=torch.float32)

        # Validate input data and target
        if torch.any(torch.isnan(patch_data)) or torch.any(torch.isinf(patch_data)):
            raise ValueError(f"Invalid patch data at index {idx}")
        if torch.any(torch.isnan(target)) or torch.any(torch.isinf(target)):
            raise ValueError(f"Invalid target at index {idx}")

        return patch_data, target

# Step 2: Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, outputs, targets):
        return self.alpha * self.mse(outputs, targets) + (1 - self.alpha) * self.smooth_l1(outputs, targets)

# Step 3: Denormalization function
def denormalize_lat_lon(norm_lat, norm_lon):
    denorm_lat = (norm_lat + 1) * 90 - 90
    denorm_lon = (norm_lon + 1) * 180 - 180
    return denorm_lat, denorm_lon

# Step 4: Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_file_path = "/home/ebuccoliero/Split_Data/simplified_interferograms.h5"
    train_csv = "/home/ebuccoliero/Split_Data/Point/ConvNeXt/train_data_grouped.csv"
    val_csv = "/home/ebuccoliero/Split_Data/Point/ConvNeXt/val_data_grouped.csv"
    test_csv = "/home/ebuccoliero/Split_Data/Point/ConvNeXt/test_data_grouped.csv"
    checkpoint_path = "/home/ebuccoliero/Split_Data/Point/ConvNeXt/Training_output/checkpoint_convnext.pth"

    # Augmentation and normalization
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.0, 0.0], std=[0.5, 0.5, 0.5, 1.0, 1.0])
    ])

    # Datasets and DataLoaders
    train_dataset = PatchDataset(h5_file_path, train_csv, transform=transform)
    val_dataset = PatchDataset(h5_file_path, val_csv, transform=transform)
    test_dataset = PatchDataset(h5_file_path, test_csv, transform=transform)
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model, optimizer, and scheduler
    model = create_model('convnext_tiny', pretrained=True, num_classes=2, in_chans=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Loss function
    criterion = CombinedLoss(alpha=0.7)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for iteration, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"NaN or Inf loss detected at iteration {iteration}")
            loss.backward()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Testing loop
    model.eval()
    predictions, ground_truth = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            ground_truth.extend(targets.cpu().numpy())

    # Denormalize predictions and ground truth
    denorm_predictions = [denormalize_lat_lon(pred[0], pred[1]) for pred in predictions]
    denorm_ground_truth = [denormalize_lat_lon(gt[0], gt[1]) for gt in ground_truth]

    # Save predictions and ground truth to CSV
    output_dir = "/home/ebuccoliero/Split_Data/Point/ConvNeXt/Training_output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "predictions_vs_ground_truth.csv")
    data = {
        "True_Latitude": [gt[0] for gt in denorm_ground_truth],
        "True_Longitude": [gt[1] for gt in denorm_ground_truth],
        "Pred_Latitude": [pred[0] for pred in denorm_predictions],
        "Pred_Longitude": [pred[1] for pred in denorm_predictions]
    }
    pd.DataFrame(data).to_csv(output_file, index=False)
    print(f"Predictions and ground truth saved to {output_file}")

    # Calculate metrics
    mae = mean_absolute_error(denorm_ground_truth, denorm_predictions)
    mse = mean_squared_error(denorm_ground_truth, denorm_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(denorm_ground_truth, denorm_predictions)

    print(f"Test Results - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    main()
