# [REPLACE sections in your original 'test.txt' file with these snippets]

# --- Imports (Ensure pandas is imported) ---
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR # <<< ADD THIS IMPORT
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights # Correct import
from PIL import Image
import pandas as pd
import os
import time
import copy
from tqdm import tqdm # Import tqdm
from torch.utils.data import Dataset
import sys
from sklearn.cluster import DBSCAN

# +++ DDP Imports +++
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse # For command-line arguments often used with DDP

# +++ DDP Helper Functions +++
def setup(rank, world_size):
    """Initializes the distributed environment using variables from torchrun."""
    # Initialize the process group. It will automatically use
    # MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE from the environment set by torchrun.
    # We still pass rank and world_size for clarity/consistency, but backend='nccl' is key.
    print(f"Rank {rank}: Attempting dist.init_process_group (backend='nccl')...") # Add print before
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank}: dist.init_process_group succeeded.") # Add print after
    torch.cuda.set_device(rank) # Pin the process to the specific GPU
    print(f"Rank {rank}/{world_size} initialized and pinned to GPU {rank}.")

def cleanup():
    """Destroys the distributed environment."""
    print(f"Rank {dist.get_rank()}: Cleaning up distributed environment.") # Get rank before destroy
    dist.destroy_process_group()
    print("Distributed environment cleaned up.")

torch.cuda.empty_cache() # Clear any cached memory

# --- Submission Generation Function (Modified for Angle Prediction) ---
def generate_angle_submission(model, val_loader, device, output_dir, roll_no, version, val_set_actual_size, val_set_expected_size):
    """
    Generates the angle prediction submission CSV file.

    Args:
        model: The trained angle prediction model (loaded with best weights).
        val_loader: DataLoader for the validation set (yields original_indices). [cite: 22]
        device: The device the model is on ('cuda' or 'cpu'). [cite: 23]
        output_dir: Directory to save the submission file. [cite: 23]
        roll_no: Student's roll number. [cite: 23]
        version: Submission version number. [cite: 23]
        val_set_actual_size: The actual number of samples processed in the validation set loader. [cite: 24]
        val_set_expected_size: The original number of rows in labels_val.csv (used for ID mapping).
    """
    print("\nGenerating angle prediction submission file...")
    model.eval() # Set model to evaluation mode

    # Store predictions mapped to their original index
    # Dictionary format: {original_index: angle}
    predictions_map = {}

    # Process validation set
    with torch.no_grad():
        # Dataloader yields (inputs, targets, original_indices)
        for inputs, _, original_indices in tqdm(val_loader, desc="Predicting on Validation Set", file=sys.stdout):
            inputs = inputs.to(device) # [cite: 26]
            outputs = model(inputs) # Shape: [batch_size, 1]

            # Move predictions to CPU and convert to numpy/list
            preds_cpu = outputs.cpu().numpy().flatten() # Flatten to get a list of angles
            indices_cpu = original_indices.cpu().numpy()

            # Store predictions mapped by their original index
            for i in range(len(inputs)): # [cite: 27]
                original_idx = indices_cpu[i]
                # --- Crucial: Ensure angle is within [0, 360] ---
                # Apply modulo 360 and handle potential negative results from modulo
                pred_angle = (preds_cpu[i] % 360.0 + 360.0) % 360.0
                # Round to nearest integer as per typical submission requirements (check if float is allowed)
                predictions_map[original_idx] = round(pred_angle)


    # --- Create DataFrame for submission ---
    num_val_predictions = len(predictions_map)
    if num_val_predictions != val_set_actual_size:
        print(f"WARNING: Number of stored predictions ({num_val_predictions}) doesn't match filtered validation set size ({val_set_actual_size}). There might be an issue.") # [cite: 29]

    # Create submission rows based on the original order (0 to N-1, where N is val_set_expected_size)
    submission_data = []
    id_counter = 0
    # --- Use val_set_expected_size for iteration to maintain original ID structure ---
    # This handles cases where some validation images might be missing
    for original_idx in range(val_set_expected_size):
        if original_idx in predictions_map:
            angle = predictions_map[original_idx]
            submission_data.append({'id': id_counter, 'angle': angle})
            id_counter += 1
        # else: # This case means the image for original_idx was missing/skipped
            # You might need to add a default value (e.g., 0) or handle as per competition rules
            # For now, we skip, assuming the competition expects predictions only for provided val images.
            # print(f"Skipping ID mapping for original index {original_idx} as it wasn't predicted (likely missing file).")

    # Check if the number of generated rows matches the count of actual predictions made
    if id_counter != num_val_predictions:
       print(f"ERROR: Mismatch between generated submission rows ({id_counter}) and number of predictions ({num_val_predictions}). Check ID mapping logic.") # [cite: 31, 32]


    df_val_preds = pd.DataFrame(submission_data)

    # Test placeholder part
    start_test_id = id_counter # Test IDs start after the last validation ID processed
    num_test_placeholders = 369 # As specified
    test_ids = list(range(start_test_id, start_test_id + num_test_placeholders))
    test_angles = [0] * num_test_placeholders # Fill with 0 as required

    df_test_placeholders = pd.DataFrame({
        'id': test_ids,
        'angle': test_angles
    })

    # Combine validation predictions and test placeholders
    df_submission = pd.concat([df_val_preds, df_test_placeholders], ignore_index=True)

    # Final check on total rows expected: number of *actual* val predictions + 369 test placeholders
    expected_total_rows = num_val_predictions + 369 # Use actual predictions count
    # --- OR check against the SPECIFICATION requirement (738 data rows + 1 header) ---
    expected_spec_rows = 738 # Total data rows required by spec
    if len(df_submission) != expected_spec_rows:
        print(f"WARNING: Final submission DataFrame has {len(df_submission)} data rows, specification requires {expected_spec_rows}.")
        print(f"       (Validation predictions: {num_val_predictions}, Test Placeholders: {num_test_placeholders})")
        # This warning is important! You might need to adjust how missing validation samples are handled
        # depending on whether the spec requires a row for *every* original validation entry, even if missing.
        # If it does, you'd need to add placeholder rows (e.g., angle=0) for the missing validation IDs.
    else:
        print(f"Submission DataFrame created with {len(df_submission)} data rows (Validation: {num_val_predictions}, Test Placeholders: {num_test_placeholders}).")


    # Save the submission file
    submission_filename = f"{roll_no}_{version}.csv" # Keep consistent naming convention
    submission_path = os.path.join(output_dir, submission_filename)
    # Ensure columns are exactly 'id', 'angle' in that order
    df_submission[['id', 'angle']].to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")


class AnglePredictionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_train=True):
        self.transform = transform
        self.img_dir = img_dir
        self.filenames = []
        self.targets = [] # To store angle
        self.original_indices = [] # Store original index for submission mapping
        self.is_train = is_train

        try:
            labels_df = pd.read_csv(csv_file)
            print(f"Processing dataset for {csv_file} in {img_dir}...") # [cite: 8]

            df_processed = labels_df.copy()

            if self.is_train:
                print("Applying data cleaning for the training dataset...")
                # --- Filter angles outside the valid range (TRAIN ONLY) ---
                df_filtered_angle = df_processed[(df_processed['angle'] >= 0) & (df_processed['angle'] <= 360)].copy()
                print(f"Filtered {len(df_processed) - len(df_filtered_angle)} rows with angles outside [0, 360].")
                df_processed = df_filtered_angle

                # --- Integrate your friend's data cleaning here (TRAIN ONLY) ---
                Q1_lat = df_processed['latitude'].quantile(0.25)
                Q3_lat = df_processed['latitude'].quantile(0.75)
                IQR_lat = Q3_lat - Q1_lat

                Q1_lon = df_processed['longitude'].quantile(0.25)
                Q3_lon = df_processed['longitude'].quantile(0.75)
                IQR_lon = Q3_lon - Q1_lon

                df_filtered_iqr = df_processed[
                    (df_processed['latitude'] >= Q1_lat - 1.5 * IQR_lat) & (df_processed['latitude'] <= Q3_lat + 1.5 * IQR_lat) &
                    (df_processed['longitude'] >= Q1_lon - 1.5 * IQR_lon) & (df_processed['longitude'] <= Q3_lon + 1.5 * IQR_lon)
                ].copy()

                lat_mean = df_filtered_iqr['latitude'].mean()
                lon_mean = df_filtered_iqr['longitude'].mean()
                df_filtered_iqr['lat_centered'] = df_filtered_iqr['latitude'] - lat_mean
                df_filtered_iqr['lon_centered'] = df_filtered_iqr['longitude'] - lon_mean

                def remove_outliers_regionwise(df, eps=50, min_samples=5):
                    cleaned = []
                    stats = []

                    for region_id in sorted(df['Region_ID'].unique()):
                        region_df = df[df['Region_ID'] == region_id].copy()
                        coords = region_df[['lon_centered', 'lat_centered']].values

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
                            'Percent_Removed': round(100 * removed / total, 2)
                        })

                        region_df = region_df[region_df['cluster'] != -1]
                        cleaned.append(region_df)

                    clean_df = pd.concat(cleaned, ignore_index=True)
                    stats_df = pd.DataFrame(stats)

                    return clean_df, stats_df

                df_cleaned, stats_df = remove_outliers_regionwise(df_filtered_iqr, eps=50, min_samples=5)

                print("=== Outlier Removal Summary ===")
                print(stats_df.to_string(index=False))
                print("\nTotal before (after angle filter):", len(df_filtered_iqr))
                print("Total after :", len(df_cleaned))
                print("Total removed:", len(df_filtered_iqr) - len(df_cleaned))
                df_processed = df_cleaned
                # --- End of data cleaning (TRAIN ONLY) ---
            else:
                print("Skipping data cleaning for the validation dataset.")

            missing_files = 0
            valid_filenames_in_order = []
            targets_in_order = []
            original_indices_in_order = []

            for idx, row in tqdm(df_processed.iterrows(), total=len(df_processed), desc="Checking images", file=sys.stdout):
                img_path = os.path.join(self.img_dir, row['filename']) # [cite: 9]
                if os.path.exists(img_path):
                    valid_filenames_in_order.append(row['filename'])
                    targets_in_order.append(torch.tensor(float(row['angle']), dtype=torch.float)) # <-- Changed to 'angle'
                    original_indices_in_order.append(row.name) # Store the index from the cleaned DataFrame
                else:
                    missing_files += 1

            self.filenames = valid_filenames_in_order
            self.targets = targets_in_order
            self.original_indices = original_indices_in_order

            if missing_files > 0:
                print(f"Warning: Skipped {missing_files} entries due to missing image files.") # [cite: 12]

            if not self.filenames:
                print(f"ERROR: No valid image files found for {csv_file}. Dataset is empty.")

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
            return None, None, -1 # [cite: 14]

        target = self.targets[idx] # Get the angle tensor
        original_index = self.original_indices[idx] # Get the original CSV index (from the cleaned DataFrame)

        if self.transform:
            image = self.transform(image)

        return image, target.unsqueeze(0), original_index # Ensure target is [1] dim for loss function


# --- Custom Loss Function: Mean Absolute Angular Error ---
def mean_absolute_angular_error_loss(outputs, targets):
    """
    Calculates the Mean Absolute Angular Error loss.
    Handles the circular nature of angles (0-360 degrees).
    Loss = mean(min(|y - y_pred|, 360 - |y - y_pred|))

    Args:
        outputs (Tensor): Model predictions, shape (batch_size, 1). Assumed to be in degrees.
        targets (Tensor): Ground truth angles, shape (batch_size, 1). Assumed to be in degrees.

    Returns:
        Tensor: A single scalar MAAE loss value.
    """
    # Ensure predictions and targets are within [0, 360) for calculation
    # We apply modulo here *within the loss calculation* for correctness,
    # but the model's raw output might not be constrained.
    outputs = outputs
    targets = targets

    # Calculate absolute difference
    abs_diff = torch.abs(outputs - targets)

    # Calculate angular distance considering circularity
    angular_dist = torch.min(abs_diff, 360.0 - abs_diff)

    # Calculate the mean over the batch
    loss = torch.mean(angular_dist)
    return loss

# --- Metric Function: MAAE (for evaluation, same calculation as loss) ---
def calculate_maae_metric(outputs, targets):
    """
    Calculates the Mean Absolute Angular Error metric (identical to the loss function).
    Used for reporting during validation.
    """
    return mean_absolute_angular_error_loss(outputs, targets)

criterion = mean_absolute_angular_error_loss


# --- Main Function for DDP ---
def main_worker(rank, world_size, args):
    setup(rank, world_size)
    print(f"Rank {rank}: Starting main worker...")

    # --- Device Setup (Specific to this rank/process) ---
    device = torch.device(f"cuda:{rank}")

    # --- Configuration (Load args) ---
    YOUR_ROLL_NO = args.roll_no
    SUBMISSION_VERSION = args.version
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    BATCH_SIZE = args.batch_size # Per-GPU batch size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    NUM_WORKERS = args.num_workers # Per-process workers

    TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'images_train')
    VAL_IMG_DIR = os.path.join(DATA_DIR, 'images_val')
    TRAIN_CSV = os.path.join(DATA_DIR, 'labels_train.csv')
    VAL_CSV = os.path.join(DATA_DIR, 'labels_val.csv')
    BEST_MODEL_PATH = f'best_angle_model_{YOUR_ROLL_NO}_v{SUBMISSION_VERSION}_rank{rank}.pth' # Rank-specific temp best? Or centralize later.
                                                                                         # Let's save only on rank 0 later.
    BEST_MODEL_PATH_FINAL = f'best_angle_model_{YOUR_ROLL_NO}_v{SUBMISSION_VERSION}.pth' # Final path saved by rank 0

    if rank == 0: # Only rank 0 creates the output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    dist.barrier() # Wait for rank 0 to create the directory if needed

    # --- Data Transforms ---
    input_size = 300 # Correct EfficientNet-B0 size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Data transforms remain the same...
    data_transforms = {
         'train': transforms.Compose([
             transforms.RandomResizedCrop(input_size),
             transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
             transforms.RandomRotation(15),
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

    # --- Create Datasets ---
    # (AnglePredictionDataset class definition remains the same)
    try:
        if rank == 0: print("\nCreating Angle Prediction Datasets...")
        # Create datasets on all ranks
        train_dataset = AnglePredictionDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=data_transforms['train'], is_train=True)
        val_dataset = AnglePredictionDataset(VAL_CSV, VAL_IMG_DIR, transform=data_transforms['val'], is_train=False)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("One or both datasets are empty after filtering.")

        VAL_SET_ACTUAL_SIZE = len(val_dataset) # This might differ slightly across ranks if filtering is not deterministic? Should be ok here.
        if rank == 0: print(f"Actual validation dataset size (before sampling): {VAL_SET_ACTUAL_SIZE}")

        # +++ Create Distributed Samplers +++
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) # Shuffle=True for training
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) # Shuffle=False for validation

        if rank == 0: print("\nCreating DataLoaders with Distributed Samplers...")
        # +++ Create DataLoaders with Samplers +++
        # Note: shuffle must be False when using a sampler
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, sampler=val_sampler)

        dataloaders = {'train': train_loader, 'val': val_loader}
        # Dataset sizes per rank (approximate for reporting)
        dataset_sizes = {x: len(dataloaders[x].dataset) // world_size for x in ['train', 'val']} # Rough estimate per rank
        if rank == 0:
             print(f"Training dataset approx size per GPU: {dataset_sizes['train']}")
             print(f"Validation dataset approx size per GPU: {dataset_sizes['val']}")
             print(f"Effective Total Batch Size: {BATCH_SIZE * world_size}")

    except Exception as e:
        print(f"Rank {rank}: Error creating Datasets/DataLoaders: {e}")
        dataloaders = None
        cleanup()
        sys.exit(1)

    # --- Model Setup ---
    if dataloaders:
        # Load model structure (same on all ranks)
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)

        # Adapt final layer (same on all ranks)
        if isinstance(model.classifier, nn.Sequential):
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, 1)
        elif isinstance(model.classifier, nn.Linear):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 1)
        else:
            if rank == 0: print("Warning: Classifier structure not recognized...")
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 1)

        if rank == 0: print("Model final layer adapted for angle prediction. Using EfficientNet-B0.")

        # +++ Move model to the correct device for this rank BEFORE wrapping +++
        model = model.to(device)

        # +++ Wrap model with DDP +++
        # find_unused_parameters can be needed if parts of the model aren't used in forward pass,
        # but usually False is fine initially.
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        if rank == 0: print(f"Model wrapped with DDP across {world_size} GPUs.")

        # Loss Function
        criterion = mean_absolute_angular_error_loss
        if rank == 0: print(f"Using Loss Function: {criterion.__name__}")

        # Optimizer
        # Use AdamW as requested previously
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        if rank == 0: print(f"Using Optimizer: AdamW with LR={LEARNING_RATE}")

        # +++ Learning Rate Scheduler +++
        # Example: Decrease LR by factor of 0.1 every 30 epochs
        LR_STEP_SIZE = 30
        LR_GAMMA = 0.1
        scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    else:
        if rank == 0: print("DataLoaders not created. Model and optimizer not initialized.")
        cleanup()
        sys.exit(1)

    # --- Training ---
    if dataloaders and 'model' in locals():
        if rank == 0: print("\nStarting DDP angle prediction training with MAAE loss...")

        # +++ Modify train_model to be DDP-aware +++
        best_model_state, history = train_model_ddp(
            rank, world_size, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
            device, NUM_EPOCHS, BEST_MODEL_PATH_FINAL
        ) # Pass rank, world_size, device, final model path

        # --- Generate Submission (Only on Rank 0) ---
        if rank == 0 and best_model_state: # Check if best_model_state is not None
            print("\nRank 0: Loading best model for submission generation...")
            # Recreate base model structure
            base_model = models.efficientnet_b0(weights=None)
            if isinstance(base_model.classifier, nn.Sequential):
                num_ftrs = base_model.classifier[-1].in_features
                base_model.classifier[-1] = nn.Linear(num_ftrs, 1)
            elif isinstance(base_model.classifier, nn.Linear):
                num_ftrs = base_model.classifier.in_features
                base_model.classifier = nn.Linear(num_ftrs, 1)
            # Load the saved state dict (remove potential 'module.' prefix if saved from DDP)
            # DDP often saves with 'module.' prefix, load_state_dict handles it by default if strict=False
            base_model.load_state_dict(best_model_state)
            base_model = base_model.to(device) # Move loaded model to rank 0's device

            # Load the original validation CSV for size check
            try:
                df_val_original = pd.read_csv(VAL_CSV)
                VAL_SET_EXPECTED_SIZE = len(df_val_original)
            except Exception as e:
                print(f"Rank 0: Error reading validation CSV for size check: {e}")
                VAL_SET_EXPECTED_SIZE = VAL_SET_ACTUAL_SIZE # Fallback
                print(f"Warning: Using actual validation size {VAL_SET_ACTUAL_SIZE} for expected size.")

            # Need a non-distributed dataloader for final validation prediction
            val_dataset_final = AnglePredictionDataset(VAL_CSV, VAL_IMG_DIR, transform=data_transforms['val'])
            val_loader_final = DataLoader(val_dataset_final, batch_size=BATCH_SIZE*2, # Can use larger batch for inference
                                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


            generate_angle_submission( # Original function should work if model is on correct device
                model=base_model,
                val_loader=val_loader_final, # Use the non-distributed loader
                device=device, # Rank 0's device
                output_dir=OUTPUT_DIR,
                roll_no=YOUR_ROLL_NO,
                version=SUBMISSION_VERSION,
                val_set_actual_size=VAL_SET_ACTUAL_SIZE, # Should match final loader size
                val_set_expected_size=VAL_SET_EXPECTED_SIZE
            )

            # Optional Plotting (only on rank 0)
            # (Plotting code remains the same, using 'history' returned from train_model_ddp)
            try:
                # ... (rest of plotting code using history dictionary) ...
                plt.savefig(os.path.join(OUTPUT_DIR, f'angle_training_plot_{YOUR_ROLL_NO}_v{SUBMISSION_VERSION}.png'))
                print(f"Rank 0: Training plot saved.")
            except ImportError:
                print("Rank 0: Matplotlib not found. Skipping plotting.")
            except Exception as e:
                print(f"Rank 0: An error occurred during plotting: {e}")

        elif rank == 0 and not best_model_state:
            print("Rank 0: Training did not produce a best model state. Submission not generated.")

    else:
        if rank == 0: print("\nTraining prerequisites not met.")

    cleanup() # Clean up process group for all ranks


# --- DDP-Aware Training Function ---
def train_model_ddp(rank, world_size, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs, best_model_save_path):
    since = time.time()
    best_model_wts_state = None # Store state_dict directly
    best_val_maae = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_maae': []}

    for epoch in range(num_epochs):
        epoch_str = f'Epoch {epoch+1}/{num_epochs}'
        if rank == 0: # Only print epoch headers on rank 0
            print(epoch_str)
            print('-' * len(epoch_str))

        # --- Set epoch for DistributedSampler - crucial for shuffling! ---
        dataloaders['train'].sampler.set_epoch(epoch)
        dataloaders['val'].sampler.set_epoch(epoch) # Set epoch for validation too

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_maae = 0.0
            total_samples = 0 # Track samples processed in this phase on this rank

            # --- Use tqdm only on rank 0 ---
            disable_tqdm = rank != 0
            data_iterator = tqdm(dataloaders[phase], desc=f"Rank {rank} {phase.capitalize()}", leave=False, file=sys.stdout, disable=disable_tqdm)

            for inputs, targets, _ in data_iterator:
                if inputs is None or targets is None: continue
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        maae_metric = calculate_maae_metric(outputs, targets)
                        # Accumulate metrics locally for this rank
                        running_maae += maae_metric.item() * inputs.size(0)

                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                # Update tqdm postfix only on rank 0 based on its own batch stats
                if rank == 0:
                    postfix = {'loss': loss.item()}
                    if phase == 'val':
                        postfix['maae'] = maae_metric.item() # Show rank 0's batch MAAE
                    data_iterator.set_postfix(postfix)

            # --- Synchronize Metrics Across All Ranks ---
            # Sum loss, maae, and samples across all ranks
            epoch_loss_tensor = torch.tensor([running_loss]).to(device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM) # Sum across ranks

            if phase == 'val':
                epoch_maae_tensor = torch.tensor([running_maae]).to(device)
                dist.all_reduce(epoch_maae_tensor, op=dist.ReduceOp.SUM)

            # Total samples across all GPUs for the epoch
            total_samples_tensor = torch.tensor([total_samples]).to(device)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

            # Calculate average metrics across all data
            # Use .item() only *after* aggregation and division
            # Avoid division by zero if total_samples_tensor is 0
            if total_samples_tensor.item() > 0:
                epoch_loss = epoch_loss_tensor.item() / total_samples_tensor.item()
                if phase == 'val':
                    epoch_maae = epoch_maae_tensor.item() / total_samples_tensor.item()
            else:
                 epoch_loss = 0.0
                 if phase == 'val':
                     epoch_maae = 0.0


            # --- Logging and Saving (Only on Rank 0) ---
            if rank == 0:
                history[f'{phase}_loss'].append(epoch_loss)
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')

                if phase == 'val':
                    history['val_maae'].append(epoch_maae)
                    print(f'Aggregated Validation MAAE: {epoch_maae:.4f}') # Print aggregated MAAE

                    if epoch_maae < best_val_maae:
                        best_val_maae = epoch_maae
                        # --- Save model state_dict (access via .module with DDP) ---
                        best_model_wts_state = copy.deepcopy(model.module.state_dict())
                        torch.save(best_model_wts_state, best_model_save_path)
                        print(f"Rank 0: Best model weights (Val MAAE: {best_val_maae:.4f}) saved to {best_model_save_path}")

        # Ensure all processes finish the epoch before starting the next
        scheduler.step() # <<< CALL SCHEDULER STEP HERE (ONCE PER EPOCH)
        dist.barrier()

    time_elapsed = time.time() - since
    if rank == 0:
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best Validation MAAE: {best_val_maae:4f}')

    # Return the best model state (only relevant for rank 0 for saving/submission) and history
    return best_model_wts_state if rank == 0 else None, history if rank == 0 else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser("EfficientNet-B2 DDP Training")
    # Add arguments matching those used in main_worker
    parser.add_argument('--roll_no', type=str, default="2022113011", help='Your roll number')
    parser.add_argument('--version', type=int, default=1, help='Submission version')
    parser.add_argument('--data_dir', type=str, default='Phase_2_data', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='submissions_angle_ddp', help='Output directory for submissions and models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size PER GPU') # Adjusted default based on single GPU test
    parser.add_argument('--epochs', type=int, default=1500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers per process')
    # Add --local-rank as an ignored argument
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank (ignored, rank determined by environment)')
    args = parser.parse_args()

    # --- Get Rank and World Size from Environment (set by srun/SLURM) ---
    # These should be set by SLURM when using srun with --ntasks-per-node
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1)) # Defaults for single process case

    print(f"Launcher: Detected Rank: {rank}, World Size: {world_size}")

    if world_size > 1 and not dist.is_available():
        raise RuntimeError("Distributed training requires 'torch.distributed' to be available.")

    if world_size > 1:
        # Call the main worker function for distributed training
        main_worker(rank, world_size, args)
    else:
        # Handle single GPU/CPU case if needed (or just run main_worker with world_size=1)
        print("Running in single-process mode (or world_size=1).")
        main_worker(rank, world_size, args) # Rank is 0, world_size is 1

    print("Main script finished.")

