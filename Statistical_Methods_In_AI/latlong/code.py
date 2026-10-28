# [REPLACE sections in your original 'test.txt' file with these snippets]

# --- Imports (Ensure pandas is imported) ---
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights # Add this line
from PIL import Image
import pandas as pd
import os
import time
import copy
from tqdm import tqdm
from torch.utils.data import Dataset
import sys
# Keep these if you plan to use the lat/long based cleaning from test2.py
from sklearn.cluster import DBSCAN
import torch.nn.functional as F  # Import functional for MSE calculation

# +++ DDP Imports +++
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse  # For command-line arguments often used with DDP


# import torch.multiprocessing as mp # No need for mp.spawn if using torchrun/srun

# +++ DDP Helper Functions +++
def setup(rank, world_size):
    """Initializes the distributed environment using variables from torchrun/srun."""
    print(f"Rank {rank}: [DEBUG] Before dist.init_process_group...")  # DEBUG PRINT
    dist.init_process_group(backend="nccl")
    print(f"Rank {rank}: [DEBUG] After dist.init_process_group.")  # DEBUG PRINT
    torch.cuda.set_device(rank)  # Pin the process to the specific GPU
    print(f"Rank {rank}/{world_size} initialized and pinned to GPU {rank}.")


def cleanup():
    """Destroys the distributed environment."""
    if dist.is_initialized():  # Check if initialized before destroying
        print(f"Rank {dist.get_rank()}: Cleaning up distributed environment.")
        dist.destroy_process_group()
        print("Distributed environment cleaned up.")


torch.cuda.empty_cache()  # Clear any cached memory


# --- Submission Generation Function (Modified for Regression with Inverse Normalization) ---
def generate_regression_submission(model, val_loader, device, output_dir, roll_no, version, val_set_actual_size,
                                   val_set_expected_size, lat_mean, lat_std, lon_mean, lon_std):
    """
    Generates the regression submission CSV file.
    Applies inverse normalization to predictions.

    Args:
        model: The trained regression model (loaded with best weights).
        val_loader: DataLoader for the validation set (yields original_indices).
        device: The device the model is on ('cuda' or 'cpu').
        output_dir: Directory to save the submission file.
        roll_no: Student's roll number.
        version: Submission version number.
        val_set_actual_size: The actual number of samples processed in the validation set loader.
        val_set_expected_size: The original number of rows in labels_val.csv (used for ID mapping).
        lat_mean (float): Mean of latitude from training data for inverse normalization.
        lat_std (float): Standard deviation of latitude from training data for inverse normalization.
        lon_mean (float): Mean of longitude from training data for inverse normalization.
        lon_std (float): Standard deviation of longitude from training data for inverse normalization.
    """
    print("\nGenerating regression submission file...")
    model.eval()  # Set model to evaluation mode

    # Store predictions mapped to their original index
    # Dictionary format: {original_index: [latitude, longitude]}
    predictions_map = {}

    # Process validation set
    with torch.no_grad():
        # Dataloader yields (inputs, targets, original_indices)
        for inputs, _, original_indices in tqdm(val_loader, desc="Predicting on Validation Set", file=sys.stdout):
            inputs = inputs.to(device)
            outputs = model(inputs)  # Shape: [batch_size, 2] (predicted lat, lon - IN NORMALIZED SPACE)

            # --- Apply Inverse Normalization ---
            # raw_prediction = (normalized_prediction * std_dev) + mean
            predicted_lat_normalized = outputs[:, 0]
            predicted_lon_normalized = outputs[:, 1]

            predicted_lat_raw = (predicted_lat_normalized * lat_std) + lat_mean
            predicted_lon_raw = (predicted_lon_normalized * lon_std) + lon_mean

            # Move inverse-normalized predictions to CPU and convert to numpy/list
            preds_cpu_raw = torch.stack([predicted_lat_raw, predicted_lon_raw], dim=1).cpu().numpy()
            indices_cpu = original_indices.cpu().numpy()

            # Store predictions mapped by their original index
            for i in range(len(inputs)):
                original_idx = indices_cpu[i]
                # Convert predictions to INTEGERS by rounding
                pred_lat = int(round(preds_cpu_raw[i, 0]))  # Round and convert Latitude
                pred_lon = int(round(preds_cpu_raw[i, 1]))  # Round and convert Longitude
                predictions_map[original_idx] = [pred_lat, pred_lon]

    # --- Create DataFrame for submission ---
    num_val_predictions = len(predictions_map)
    if num_val_predictions != val_set_actual_size:
        print(
            f"WARNING: Number of stored predictions ({num_val_predictions}) doesn't match filtered validation set size ({val_set_actual_size}). There might be an issue.")

    # Create submission rows based on the original order (0 to N-1, where N is val_set_expected_size)
    submission_data = []
    id_counter = 0
    for original_idx in range(
            val_set_expected_size):  # Iterate through potential original indices from the full validation set
        if original_idx in predictions_map:
            lat, lon = predictions_map[original_idx]
            submission_data.append({'id': id_counter, 'Latitude': lat, 'Longitude': lon})
            id_counter += 1
        # else: # This case means the image for original_idx was missing/skipped
        # Depending on competition rules, you might need to add a placeholder row here
        # print(f"Skipping ID mapping for original index {original_idx} as it wasn't predicted (likely missing file).") # Optional debug

    # Check if the number of generated rows matches the count of predictions
    if id_counter != num_val_predictions:
        print(
            f"ERROR: Mismatch between generated submission rows ({id_counter}) and number of predictions ({num_val_predictions}). Check ID mapping logic.")

    df_val_preds = pd.DataFrame(submission_data)

    # Test placeholder part (IDs start after the last validation ID)
    start_test_id = id_counter  # Start test IDs after the last valid validation ID
    # As per spec: "remaining 369 rows are for the test set"
    num_test_placeholders = 369
    test_ids = list(range(start_test_id, start_test_id + num_test_placeholders))
    test_latitudes = [0] * num_test_placeholders  # Fill with 0
    test_longitudes = [0] * num_test_placeholders  # Fill with 0

    df_test_placeholders = pd.DataFrame({
        'id': test_ids,
        'Latitude': test_latitudes,
        'Longitude': test_longitudes
    })

    # Combine validation predictions and test placeholders
    df_submission = pd.concat([df_val_preds, df_test_placeholders], ignore_index=True)

    # Final check on total rows expected: number of valid val samples + 369 test placeholders
    expected_spec_rows = 738  # Total data rows required by spec
    if len(df_submission) != expected_spec_rows:
        print(
            f"WARNING: Final submission DataFrame has {len(df_submission)} data rows, specification requires {expected_spec_rows}.")
        print(f"       (Validation predictions: {num_val_predictions}, Test Placeholders: {num_test_placeholders})")
        # This might happen if the number of *valid* validation images is not 738 - 369 = 369.
        # The code builds the submission based on the valid images found.
    else:
        print(
            f"Submission DataFrame created with {len(df_submission)} data rows (Validation: {num_val_predictions}, Test Placeholders: {num_test_placeholders}).")

    # Save the submission file
    submission_filename = f"{roll_no}_{version}.csv"
    submission_path = os.path.join(output_dir, submission_filename)
    # Ensure columns are exactly 'id', 'Latitude', 'Longitude' in that order
    df_submission[['id', 'Latitude', 'Longitude']].to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")


# --- LatLongRegressionDataset Class (Modified for Normalization) ---
class LatLongRegressionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_train=True, lat_mean=0.0, lat_std=1.0, lon_mean=0.0,
                 lon_std=1.0):
        self.transform = transform
        self.img_dir = img_dir
        self.filenames = []
        # We only need to store normalized targets for training
        self.targets_normalized = []  # Store normalized [latitude, longitude]
        self.original_indices = []  # Store original index for submission mapping
        self.is_train = is_train

        # Store the normalization stats
        self.lat_mean = lat_mean
        self.lat_std = lat_std
        self.lon_mean = lon_mean
        self.lon_std = lon_std

        try:
            labels_df = pd.read_csv(csv_file)
            print(f"Processing dataset for {csv_file} in {img_dir}...")

            df_processed = labels_df.copy()

            if self.is_train:
                print("Applying data cleaning for the training dataset...")

                print("Applying data cleaning for the training dataset...")

                # List of filenames with abnormal locations to remove
                abnormal_filenames = [
                    'img_1618.jpg', 'img_1632.jpg', 'img_2233.jpg', 'img_2538.jpg', 'img_2558.jpg',
                    'img_2717.jpg', 'img_2726.jpg', 'img_2727.jpg', 'img_2744.jpg', 'img_2750.jpg',
                    'img_2761.jpg', 'img_2762.jpg', 'img_2764.jpg', 'img_2766.jpg', 'img_2767.jpg',
                    'img_2768.jpg', 'img_2769.jpg', 'img_2770.jpg', 'img_2771.jpg', 'img_2772.jpg',
                    'img_2773.jpg', 'img_2774.jpg', 'img_2775.jpg', 'img_2776.jpg', 'img_2777.jpg',
                    'img_2778.jpg', 'img_2779.jpg', 'img_2780.jpg', 'img_2781.jpg', 'img_2782.jpg',
                    'img_2783.jpg', 'img_2784.jpg', 'img_2785.jpg', 'img_2786.jpg', 'img_2787.jpg',
                    'img_2788.jpg', 'img_2789.jpg', 'img_2790.jpg', 'img_2791.jpg', 'img_2792.jpg',
                    'img_2793.jpg', 'img_2794.jpg', 'img_2795.jpg', 'img_2796.jpg', 'img_2797.jpg',
                    'img_2798.jpg', 'img_2799.jpg', 'img_2800.jpg', 'img_2801.jpg', 'img_2802.jpg',
                    'img_2803.jpg', 'img_2804.jpg', 'img_2805.jpg', 'img_2806.jpg', 'img_2807.jpg',
                    'img_2808.jpg', 'img_2809.jpg', 'img_2810.jpg', 'img_2811.jpg', 'img_2812.jpg',
                    'img_2813.jpg', 'img_2814.jpg', 'img_3619.jpeg', 'img_3690.jpg', 'img_3691.jpg',
                    'img_3713.jpg', 'img_3729.jpg', 'img_3731.jpg', 'img_4516.jpg', 'img_4523.jpg',
                    'img_4547.jpg', 'img_4553.jpg', 'img_4560.jpg', 'img_4632.png', 'img_6115.jpeg'
                ]

                # Remove rows with abnormal locations
                df_processed = df_processed[~df_processed['filename'].isin(abnormal_filenames)]

                # --- Filter angles outside the valid range (TRAIN ONLY) ---
                df_filtered_angle = df_processed[(df_processed['angle'] >= 0) & (df_processed['angle'] <= 360)].copy()
                print(f"Filtered {len(df_processed) - len(df_filtered_angle)} rows with angles outside [0, 360].")
                df_processed = df_filtered_angle

                # Keep this cleaning if you want to remove outliers from training data
                Q1_lat = df_processed['latitude'].quantile(0.25)
                Q3_lat = df_processed['latitude'].quantile(0.75)
                IQR_lat = Q3_lat - Q1_lat

                Q1_lon = df_processed['longitude'].quantile(0.25)
                Q3_lon = df_processed['longitude'].quantile(0.75)
                IQR_lon = Q3_lon - Q1_lon

                df_filtered_iqr = df_processed[
                    (df_processed['latitude'] >= Q1_lat - 1.5 * IQR_lat) & (
                            df_processed['latitude'] <= Q3_lat + 1.5 * IQR_lat) &
                    (df_processed['longitude'] >= Q1_lon - 1.5 * IQR_lon) & (
                            df_processed['longitude'] <= Q3_lon + 1.5 * IQR_lon)
                    ].copy()

                lat_mean_filtered = df_filtered_iqr['latitude'].mean()
                lon_mean_filtered = df_filtered_iqr['longitude'].mean()
                df_filtered_iqr['lat_centered'] = df_filtered_iqr['latitude'] - lat_mean_filtered
                df_filtered_iqr['lon_centered'] = df_filtered_iqr['longitude'] - lon_mean_filtered

                def remove_outliers_regionwise(df, eps=50, min_samples=5):
                    cleaned = []
                    stats = []

                    if 'Region_ID' not in df.columns:
                        print(
                            "Warning: 'Region_ID' column not found for region-wise DBSCAN. Performing DBSCAN on entire dataset.")
                        # Fallback to DBSCAN on the entire dataset if Region_ID is missing
                        coords = df[['lon_centered', 'lat_centered']].values
                        if len(coords) > 0:
                            db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
                            df['cluster'] = db.labels_
                            cleaned.append(df[df['cluster'] != -1].copy())
                            stats.append({
                                'Region_ID': 'All',
                                'Original': len(df),
                                'Kept': (df['cluster'] != -1).sum(),
                                'Removed': (df['cluster'] == -1).sum(),
                                'Percent_Removed': round(
                                    100 * (df['cluster'] == -1).sum() / len(df) if len(df) > 0 else 0, 2)
                            })
                        clean_df = pd.concat(cleaned, ignore_index=True) if cleaned else pd.DataFrame()
                        stats_df = pd.DataFrame(stats)
                        return clean_df, stats_df

                    for region_id in sorted(df['Region_ID'].unique()):
                        region_df = df[df['Region_ID'] == region_id].copy()
                        coords = region_df[['lon_centered', 'lat_centered']].values

                        if len(coords) > 0:  # Ensure there are samples in the region
                            db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
                            region_df['cluster'] = db.labels_

                            total = len(region_df)
                            kept = (region_df['cluster'] != -1).sum()
                            removed = (region_df['cluster'] == -1).sum()

                            stats.append({
                                'Region_ID': region_id,
                                'Original': total,
                                'Kept': kept,
                                'Removed': removed,
                                'Percent_Removed': round(100 * removed / total, 2) if total > 0 else 0
                            })

                            region_df = region_df[region_df['cluster'] != -1]
                            cleaned.append(region_df)
                        else:
                            stats.append({
                                'Region_ID': region_id,
                                'Original': 0, 'Kept': 0, 'Removed': 0, 'Percent_Removed': 0
                            })

                    clean_df = pd.concat(cleaned, ignore_index=True) if cleaned else pd.DataFrame()
                    stats_df = pd.DataFrame(stats)

                    return clean_df, stats_df

                df_cleaned, stats_df = remove_outliers_regionwise(df_filtered_iqr, eps=50, min_samples=5)
                df_processed = df_cleaned
                # --- End of data cleaning (TRAIN ONLY) ---
            else:
                # For validation and submission, we use the original data rows that have images.
                print("Skipping training data cleaning for the validation dataset.")
                # No extra cleaning needed for validation data rows here, just filter by image existence below.

            self.filenames = []
            self.targets_normalized = []
            self.original_indices = []  # This should map back to the ORIGINAL labels_df index

            missing_files = 0
            # Iterate over the *original* labels_df to maintain consistent original_indices
            # Check for image existence and if the index was kept after cleaning (if is_train)
            for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df),
                                 desc="Checking images and applying filters", file=sys.stdout):
                img_path = os.path.join(self.img_dir, row['filename'])
                # Check if image exists AND if this row is kept after cleaning (if is_train)
                if os.path.exists(img_path):
                    # For is_train, check if the original index exists in the cleaned df.
                    # For not is_train (validation), we keep all rows with existing images.
                    if not self.is_train or (self.is_train and idx in df_processed.index):
                        self.filenames.append(row['filename'])

                        raw_lat = float(row['latitude'])
                        raw_lon = float(row['longitude'])

                        # Normalize and store normalized latitude and longitude
                        normalized_lat = (raw_lat - self.lat_mean) / self.lat_std if self.lat_std != 0 else raw_lat
                        normalized_lon = (raw_lon - self.lon_mean) / self.lon_std if self.lon_std != 0 else raw_lon
                        self.targets_normalized.append(
                            torch.tensor([normalized_lat, normalized_lon], dtype=torch.float))

                        self.original_indices.append(idx)  # Store the original index

                    # If is_train and the original index is NOT in df_processed.index, it means it was cleaned out - we skip it.
                    # If not is_train, we keep all rows with images that exist.
                else:
                    missing_files += 1

            if missing_files > 0:
                print(f"Warning: Skipped {missing_files} entries due to missing image files.")

            if not self.filenames:
                print(f"ERROR: No valid image files found for {csv_file}. Dataset is empty.")
            else:
                print(f"Successfully loaded {len(self.filenames)} valid images for {csv_file}.")


        except Exception as e:
            print(f"Error loading or processing CSV {csv_file}: {e}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.filenames[idx])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_name}: {e}. Returning None.")
            return None, None, -1  # Return None image, None target, invalid index

        # Return the normalized target for training/validation loss calculation
        target = self.targets_normalized[idx]  # Get the normalized [latitude, longitude] tensor
        original_index = self.original_indices[idx]  # Get the original CSV index

        if self.transform:
            image = self.transform(image)

        # Return image, normalized target tensor [lat, lon], and original index
        return image, target, original_index


# --- Custom Loss Function: Average Lat/Lon MSE ---
# This function works correctly with normalized targets
def average_lat_lon_mse_loss(outputs, targets):
    """
    Calculates the average MSE loss for latitude and longitude separately.
    Loss = 0.5 * (MSE_lat + MSE_lon)
    Args:
        outputs (Tensor): Model predictions, shape (batch_size, 2) -> [lat, lon] (normalized)
        targets (Tensor): Ground truth, shape (batch_size, 2) -> [lat, lon] (normalized)
    Returns:
        Tensor: A single scalar loss value.
    """
    # Ensure outputs and targets have the expected shape
    if outputs.shape[1] != 2 or targets.shape[1] != 2:
        raise ValueError(
            f"Expected outputs and targets to have 2 columns (lat, lon), but got shapes {outputs.shape} and {targets.shape}")

    # Calculate MSE for latitude (column 0)
    loss_lat = F.mse_loss(outputs[:, 0], targets[:, 0])
    # Calculate MSE for longitude (column 1)
    loss_lon = F.mse_loss(outputs[:, 1], targets[:, 1])

    # Calculate the average
    average_loss = 0.5 * (loss_lat + loss_lon)
    return average_loss


def unnormalize(tensor, mean, std):
    return tensor * std + mean


# --- DDP-Aware Training Function (Modified for Regression and Normalization) ---
def train_model_ddp_regression(rank, world_size, model, criterion, optimizer, scheduler, dataloaders, device,
                               num_epochs, best_model_save_path, lat_mean, lat_std, lon_mean, lon_std):
    since = time.time()
    best_model_wts_state = None  # Store state_dict directly
    # Track best validation loss (lower is better for regression)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}  # Only tracking loss for regression (on normalized targets)

    for epoch in range(num_epochs):
        epoch_str = f'Epoch {epoch + 1}/{num_epochs}'
        if rank == 0:
            print(epoch_str)
            print('-' * len(epoch_str))

        # --- Set epoch for DistributedSampler ---
        dataloaders['train'].sampler.set_epoch(epoch)
        # Setting epoch for val sampler is generally not necessary as validation is usually not shuffled per epoch,
        # but it doesn't hurt and can ensure consistency if needed.
        if dataloaders['val'].sampler:
            dataloaders['val'].sampler.set_epoch(epoch)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()  # Clear cache before validation

            running_loss = 0.0
            total_real_loss = 0.0
            total_samples = 0  # Track samples processed in this phase on this rank

            # --- Use tqdm only on rank 0 ---
            disable_tqdm = rank != 0
            data_iterator = tqdm(dataloaders[phase], desc=f"Rank {rank} {phase.capitalize()}", leave=False,
                                 file=sys.stdout, disable=disable_tqdm)

            # dataloader yields (inputs, targets [lat, lon] - NORMALIZED, original_indices)
            for inputs, targets, _ in data_iterator:
                if inputs is None or targets is None: continue
                inputs = inputs.to(device)
                targets = targets.to(device)  # Targets are [lat, lon] tensor (NORMALIZED)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Get predictions [pred_lat, pred_lon] (in NORMALIZED space)
                    loss = criterion(outputs, targets)  # Use the average MSE loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                # Unnormalize predictions for real-scale loss tracking (optional, but good for monitoring)
                outputs_lat_real = unnormalize(outputs.detach()[:, 0], lat_mean, lat_std)
                outputs_lon_real = unnormalize(outputs.detach()[:, 1], lon_mean, lon_std)
                targets_lat_real = unnormalize(targets[:, 0], lat_mean, lat_std)
                targets_lon_real = unnormalize(targets[:, 1], lon_mean, lon_std)

                # Calculate real-scale MSE for monitoring (not used for backward pass)
                real_loss_lat = F.mse_loss(outputs_lat_real, targets_lat_real)
                real_loss_lon = F.mse_loss(outputs_lon_real, targets_lon_real)
                total_real_loss += (real_loss_lat.item() + real_loss_lon.item()) / 2.0  # Average real loss

                # --- Update tqdm description with current loss ---
                if rank == 0:
                    data_iterator.set_postfix(loss=loss.item())

            # --- Synchronize Metrics Across All Ranks ---
            epoch_loss_tensor = torch.tensor([running_loss]).to(device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)

            total_samples_tensor = torch.tensor([total_samples]).to(device)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

            total_real_loss_tensor = torch.tensor([total_real_loss]).to(device)
            dist.all_reduce(total_real_loss_tensor, op=dist.ReduceOp.SUM)

            # Calculate average loss across all data
            if total_samples_tensor.item() > 0:
                epoch_loss = epoch_loss_tensor.item() / total_samples_tensor.item()
                avg_real_loss = total_real_loss_tensor.item() / total_samples_tensor.item()
            else:
                epoch_loss = 0.0
                avg_real_loss = 0.0

            # --- Logging and Saving (Only on Rank 0) ---
            if rank == 0:
                history[f'{phase}_loss'].append(epoch_loss)
                print(f'Aggregated {phase.capitalize()} Loss: {epoch_loss:.4f}, Average Loss: {avg_real_loss:.4f}')

                # Save best model based on minimum validation loss
                if phase == 'val' and epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    # Save model state_dict (access via .module with DDP)
                    best_model_wts_state = copy.deepcopy(model.module.state_dict())
                    torch.save(best_model_wts_state, best_model_save_path)
                    print(f"Rank 0: Best model weights (Val Loss: {best_val_loss:.4f}) saved to {best_model_save_path}")

        scheduler.step()
        dist.barrier()  # Ensure all processes finish the epoch before starting the next

    time_elapsed = time.time() - since
    if rank == 0:
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best Validation Loss: {best_val_loss:4f}')

    # Return the best model state (only relevant for rank 0 for saving/submission) and history
    return best_model_wts_state if rank == 0 else None, history if rank == 0 else None


# --- Main Function for DDP ---
# Updated main_worker signature to accept args
def main_worker(rank, world_size, args):
    # --- Setup DDP first ---
    setup(rank, world_size)
    print(f"Rank {rank}: Starting main worker...")

    # --- Device Setup (Specific to this rank/process) ---
    device = torch.device(f"cuda:{rank}")

    # --- Configuration (Load args) ---
    YOUR_ROLL_NO = args.roll_no
    SUBMISSION_VERSION = args.version
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    BATCH_SIZE = args.batch_size  # Per-GPU batch size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    NUM_WORKERS = args.num_workers  # Per-process workers

    TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'images_train')
    VAL_IMG_DIR = os.path.join(DATA_DIR, 'images_val')
    TRAIN_CSV = os.path.join(DATA_DIR, 'labels_train.csv')
    VAL_CSV = os.path.join(DATA_DIR, 'labels_val.csv')
    BEST_MODEL_PATH_FINAL = os.path.join(OUTPUT_DIR,
                                         f'best_regression_model_{YOUR_ROLL_NO}_v{SUBMISSION_VERSION}.pth')  # Final path saved by rank 0

    if rank == 0:  # Only rank 0 creates the output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    dist.barrier()  # Wait for rank 0 to create the directory if needed

    # --- Calculate and Broadcast Normalization Stats (Mean and Std Dev) ---
    # Calculate on Rank 0 using pandas on the *full* training CSV *after* DDP setup
    # This ensures all ranks use the same normalization values derived from the entire training distribution.
    lat_mean, lat_std, lon_mean, lon_std = 0.0, 1.0, 0.0, 1.0  # Initialize with defaults

    if rank == 0:
        try:
            print(f"Rank {rank}: Calculating normalization stats from training data...")
            df_train_full = pd.read_csv(TRAIN_CSV)
            # Calculate on the raw data for a more general normalization.
            # If using cleaning before stats, the cleaning logic needs to be applied here first
            # *before* calculating mean/std. For now, calculating on raw.
            lat_mean = df_train_full['latitude'].mean()
            lat_std = df_train_full['latitude'].std()
            lon_mean = df_train_full['longitude'].mean()
            lon_std = df_train_full['longitude'].std()
            # Handle cases where std could be 0
            if lat_std == 0: lat_std = 1.0
            if lon_std == 0: lon_std = 1.0

            print(f"Rank {rank}: Calculated Training Lat/Lon Mean: ({lat_mean:.2f}, {lon_mean:.2f})")
            print(f"Rank {rank}: Calculated Training Lat/Lon Std Dev: ({lat_std:.2f}, {lon_std:.2f})")

            # Package stats into a tensor for broadcasting
            stats_tensor = torch.tensor([lat_mean, lat_std, lon_mean, lon_std], dtype=torch.float).to(
                device)  # Move to device for broadcast
        except Exception as e:
            print(f"Rank {rank}: Error calculating normalization stats: {e}. Using defaults (0, 1).")
            # Use default 0, 1 and create tensor on device
            stats_tensor = torch.tensor([lat_mean, lat_std, lon_mean, lon_std], dtype=torch.float).to(device)
    else:
        # Create an empty tensor on other ranks to receive the broadcast on their device
        stats_tensor = torch.empty(4, dtype=torch.float).to(device)

    # Broadcast the stats tensor from rank 0 to all other ranks *after* setup
    print(f"Rank {rank}: [DEBUG] Before dist.broadcast of stats_tensor...")  # DEBUG PRINT
    dist.broadcast(stats_tensor, src=0)
    print(f"Rank {rank}: [DEBUG] After dist.broadcast of stats_tensor.")  # DEBUG PRINT

    # Unpack the received stats on all ranks
    lat_mean, lat_std, lon_mean, lon_std = stats_tensor.tolist()

    if rank != 0:  # Print received stats on other ranks
        print(f"Rank {rank}: Received Normalization Stats Lat/Lon Mean: ({lat_mean:.2f}, {lon_mean:.2f})")
        print(f"Rank {rank}: Received Normalization Stats Lat/Lon Std Dev: ({lat_std:.2f}, {lon_std:.2f})")

    input_size = 224

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize
        ]),
    }

    # --- Create Datasets (Pass normalization stats) ---
    try:
        if rank == 0: print("\nCreating Regression Datasets...")
        # Pass the calculated/broadcasted normalization stats to the dataset
        train_dataset = LatLongRegressionDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=data_transforms['train'],
                                                 is_train=True,
                                                 lat_mean=lat_mean, lat_std=lat_std, lon_mean=lon_mean, lon_std=lon_std)
        val_dataset = LatLongRegressionDataset(VAL_CSV, VAL_IMG_DIR, transform=data_transforms['val'], is_train=False,
                                               lat_mean=lat_mean, lat_std=lat_std, lon_mean=lon_mean, lon_std=lon_std)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("One or both datasets are empty after filtering.")

        # Get the expected size of the validation set from the original CSV on rank 0
        VAL_SET_EXPECTED_SIZE = 0
        if rank == 0:
            try:
                df_val_original = pd.read_csv(VAL_CSV)
                VAL_SET_EXPECTED_SIZE = len(df_val_original)
                print(f"Rank 0: Original validation set size (from CSV): {VAL_SET_EXPECTED_SIZE}")
            except Exception as e:
                print(f"Rank 0: Error reading validation CSV for expected size: {e}")
                # This should ideally not happen if the initial CSV read for dataset creation succeeded.

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        if rank == 0: print("\nCreating DataLoaders with Distributed Samplers...")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                  pin_memory=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=True, sampler=val_sampler)

        dataloaders = {'train': train_loader, 'val': val_loader}
        # Dataset sizes used for aggregated loss calculation
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
        if rank == 0:
            print(f"Training dataset total size (after filtering): {dataset_sizes['train']}")
            print(f"Validation dataset total size (after filtering): {dataset_sizes['val']}")
            print(f"Effective Total Batch Size (across all GPUs): {BATCH_SIZE * world_size}")


    except Exception as e:
        print(f"Rank {rank}: Error creating Datasets/DataLoaders: {e}")
        dataloaders = None
        cleanup()
        sys.exit(1)

    # --- Model Setup ---
    if dataloaders:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 # Add this line
        model = models.efficientnet_b0(weights=weights) # Add this line

        # Modify the classifier layer for EfficientNet-B0 (output size: 2 for lat/lon)
        num_ftrs = model.classifier[1].in_features # EfficientNet classifier is Sequential, the last layer is Linear
        model.classifier[1] = nn.Linear(num_ftrs, 2) # Replace the last layer

        if rank == 0: print("Model final layer adapted for regression (output size: 2).")

        model = model.to(device)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        if rank == 0: print(f"Model wrapped with DDP across {world_size} GPUs.")

        criterion = average_lat_lon_mse_loss  # Use the custom average MSE loss on normalized targets
        if rank == 0: print(f"Using Loss Function: {criterion.__name__}")

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        if rank == 0: print(f"Using Optimizer: AdamW with LR={LEARNING_RATE}")

        LR_STEP_SIZE = 30
        LR_GAMMA = 0.1
        scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    else:
        if rank == 0: print("DataLoaders not created. Model and optimizer not initialized.")
        cleanup()
        sys.exit(1)

    # --- Training ---
    if dataloaders and 'model' in locals():
        if rank == 0: print("\nStarting DDP lat/lon regression training with average MSE loss...")

        best_model_state, history = train_model_ddp_regression(
            rank, world_size, model, criterion, optimizer, scheduler, dataloaders,
            device, NUM_EPOCHS, BEST_MODEL_PATH_FINAL,
            lat_mean=lat_mean, lat_std=lat_std, lon_mean=lon_mean, lon_std=lon_std
        )

        # --- Generate Submission (Only on Rank 0) ---
        if rank == 0 and best_model_state:
            print("\nRank 0: Loading best model for submission generation...")
            base_model = models.efficientnet_b0(weights=None)
            num_ftrs = base_model.fc.in_features
            base_model.fc = nn.Linear(num_ftrs, 2)
            base_model.load_state_dict(best_model_state)
            base_model = base_model.to(device)

            # Need a non-distributed dataloader for final validation prediction
            # Pass the normalization stats again here so the dataset doesn't try to calculate them (is_train=False)
            # and the submission function has access to them.
            val_dataset_final = LatLongRegressionDataset(VAL_CSV, VAL_IMG_DIR, transform=data_transforms['val'],
                                                         is_train=False,
                                                         lat_mean=lat_mean, lat_std=lat_std, lon_mean=lon_mean,
                                                         lon_std=lon_std)
            val_loader_final = DataLoader(val_dataset_final, batch_size=BATCH_SIZE * 2,
                                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

            val_set_actual_size_for_submission = len(val_loader_final.dataset)

            # Get the expected size of the validation set from the original CSV
            VAL_SET_EXPECTED_SIZE = 0
            try:
                df_val_original = pd.read_csv(VAL_CSV)
                VAL_SET_EXPECTED_SIZE = len(df_val_original)
            except Exception as e:
                print(f"Rank 0: Error reading validation CSV for expected size (second time): {e}")
                VAL_SET_EXPECTED_SIZE = val_set_actual_size_for_submission  # Fallback

            generate_regression_submission(
                model=base_model,
                val_loader=val_loader_final,
                device=device,
                output_dir=OUTPUT_DIR,
                roll_no=YOUR_ROLL_NO,
                version=SUBMISSION_VERSION,
                val_set_actual_size=val_set_actual_size_for_submission,
                val_set_expected_size=VAL_SET_EXPECTED_SIZE,
                # Pass normalization stats to the submission function
                lat_mean=lat_mean, lat_std=lat_std, lon_mean=lon_mean, lon_std=lon_std
            )

            # Optional Plotting (only on rank 0)
            try:
                plt.figure(figsize=(8, 5))
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Validation Loss')
                plt.title('Regression Loss vs. Epochs')
                plt.xlabel('Epochs')
                plt.ylabel('Loss (Avg Lat/Lon MSE - Normalized)')  # Clarify label
                plt.legend()

                plt.tight_layout()
                plot_filename = f'regression_training_plot_{YOUR_ROLL_NO}_v{SUBMISSION_VERSION}.png'
                plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
                print(f"Rank 0: Training plot saved to {os.path.join(OUTPUT_DIR, plot_filename)}")
            except ImportError:
                print("Rank 0: Matplotlib not found. Skipping plotting.")
            except Exception as e:
                print(f"Rank 0: An error occurred during plotting: {e}")

        elif rank == 0 and not best_model_state:
            print("Rank 0: Training did not produce a best model state. Submission not generated.")

    else:
        if rank == 0: print("\nTraining prerequisites not met.")

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("EfficientNet_B0 DDP Lat/Lon Regression Training")
    parser.add_argument('--roll_no', type=str, default="2022113011", help='Your roll number')
    parser.add_argument('--version', type=int, default=1, help='Submission version')
    parser.add_argument('--data_dir', type=str, default='Phase_2_data', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='submissions_regression_ddp',
                        help='Output directory for submissions and models')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size PER GPU')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers per process')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank (ignored, rank determined by environment)')  # Keep for torchrun compatibility
    args = parser.parse_args()

    # --- Get Rank and World Size from Environment (set by srun/SLURM/torchrun) ---
    # These environment variables are typically set automatically by the DDP launcher (torchrun, srun).
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Launcher: Detected Rank: {rank}, World Size: {world_size}")

    if world_size > 1 and not dist.is_available():
        raise RuntimeError("Distributed training requires 'torch.distributed' to be available.")

    # Call the main worker function. The DDP setup is handled inside main_worker.
    # The stats calculation and broadcast is also handled inside main_worker after setup.
    main_worker(rank, world_size, args)

    print("Main script finished.")
