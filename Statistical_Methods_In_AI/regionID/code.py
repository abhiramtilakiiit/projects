import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd
import os
import time
import copy
import sys
from tqdm import tqdm # Import tqdm

# +++ DDP Imports +++
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse # For command-line arguments often used with DDP

# +++ DDP Helper Functions +++
def setup(rank, world_size):
    """Initializes the distributed environment."""
    # --- Set environment variables for DDP ---
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358' # <<< CHANGE THIS PORT if it's in use

    print(f"Rank {rank}: Attempting dist.init_process_group (backend='nccl')...") # Add print before
    # Using GLOO for CPU or mixed CPU/GPU, NCCL for multi-GPU
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        print(f"Rank {rank}: dist.init_process_group succeeded with backend='{backend}'.") # Add print after
        if torch.cuda.is_available():
            torch.cuda.set_device(rank) # Pin the process to the specific GPU
            print(f"Rank {rank}/{world_size} initialized and pinned to GPU {rank}.")
        else:
            print(f"Rank {rank}/{world_size} initialized using CPU.")
    except Exception as e:
        print(f"Rank {rank}: Error during dist.init_process_group: {e}")
        raise # Re-raise the exception after printing

def cleanup():
    """Destroys the distributed environment."""
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: Cleaning up distributed environment.")
        dist.destroy_process_group()
        print("Distributed environment cleaned up.")


# --- Configuration ---
# <<<--- SET THESE VALUES --->>>
YOUR_ROLL_NO = "2022113011" # Replace with your actual roll number
SUBMISSION_VERSION = 1      # Increment this for new submissions
# <<<------------------------>>>

DATA_DIR = 'Phase_2_data'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'images_train')
VAL_IMG_DIR = os.path.join(DATA_DIR, 'images_val')
TRAIN_CSV = os.path.join(DATA_DIR, 'labels_train.csv')
VAL_CSV = os.path.join(DATA_DIR, 'labels_val.csv')
OUTPUT_DIR = 'submissions_region_ddp' # Directory to save submission files (changed for DDP)
# BEST_MODEL_PATH will be constructed later using OUTPUT_DIR


# --- Argument Parsing for DDP ---
parser = argparse.ArgumentParser("Resnet18 DDP Classification Training")
parser.add_argument('--roll_no', type=str, default=YOUR_ROLL_NO, help='Your roll number')
parser.add_argument('--version', type=int, default=SUBMISSION_VERSION, help='Submission version')
parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to data directory')
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Output directory for submissions and models')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size PER GPU') # Batch size per GPU
parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers per process')
# Add --local-rank as an ignored argument (set by torchrun/srun)
parser.add_argument('--local-rank', type=int, default=0, help='Local rank (ignored, rank determined by environment)')
args = parser.parse_args()

# Update configurations from parsed arguments
YOUR_ROLL_NO = args.roll_no
SUBMISSION_VERSION = args.version
DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.lr
NUM_WORKERS = args.num_workers

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, f'best_model_region_{YOUR_ROLL_NO}_v{SUBMISSION_VERSION}.pth') # Save to output dir

# --- Get Rank and World Size from Environment (set by torchrun/srun) ---
# These should be set by the DDP launcher
# Moved here to be accessible before setup
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 4)) # Defaults for single process case


# --- Setup DDP (Initialization is now here and unconditional) ---
# Setup must be called regardless of world_size to initialize the process group
setup(rank, world_size)
# Ensure all processes are set up before proceeding
dist.barrier()

torch.cuda.empty_cache() # Clear any cached memory after setup

# --- Custom Dataset (Modified to potentially return None for missing images and store original index) ---
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, label_map=None):
        self.transform = transform
        self.img_dir = img_dir
        self.label_map = label_map
        self.data = [] # Store (filename, label, original_index) tuples

        try:
            labels_df = pd.read_csv(csv_file)

            # Pre-filter for existing images and apply mapping
            if rank == 0: print(f"Rank {rank}: Filtering dataset for existing images in {img_dir}...")
            missing_files = 0
            # Iterate through original dataframe rows to maintain original order for validation submission mapping
            for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"Rank {rank} Checking images", leave=False, file=sys.stdout, disable=(rank != 0)):
                img_path = os.path.join(self.img_dir, row['filename'])
                # For training, apply filtering and mapping
                if 'Region_ID' in row and pd.notna(row['Region_ID']):
                    original_region_id = int(row['Region_ID'])
                    if self.label_map is not None and original_region_id in self.label_map:
                        if os.path.exists(img_path):
                            self.data.append((row['filename'], self.label_map[original_region_id], idx)) # Store filename, mapped label, original index
                        else:
                            # if rank == 0: print(f"Warning: Image file not found {img_path}. Skipping.")
                            missing_files += 1
                    else:
                        # if rank == 0: print(f"Warning: Region_ID {original_region_id} from {row['filename']} not found in training label map. Skipping.")
                        missing_files += 1
                # For validation (assuming no Region_ID column or it's not used for filtering beyond file existence)
                # Modify this logic if validation set requires different filtering/mapping
                elif 'filename' in row and os.path.exists(img_path):
                     # For validation, we just need the image and the original index for submission
                     # Use a placeholder label like -1 as it won't be used for loss/accuracy
                     self.data.append((row['filename'], -1, idx))
                else:
                    missing_files += 1


            if missing_files > 0 and rank == 0:
                 print(f"Rank {rank}: Warning: Skipped {missing_files} entries due to missing image files or unmappable Region_IDs.")

            if not self.data and rank == 0:
                 print(f"Rank {rank}: ERROR: No valid data entries found for {csv_file}. Dataset is empty.")

        except FileNotFoundError:
            if rank == 0: print(f"Rank {rank}: ERROR: CSV file not found: {csv_file}")
        except Exception as e:
            if rank == 0: print(f"Rank {rank}: Error loading or processing CSV {csv_file}: {e}")
            # Keep lists empty


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename, label, original_index = self.data[idx]
        img_name = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            if rank == 0: print(f"Rank {rank}: Error opening image {img_name}: {e}. Returning None.")
            # Return None for image, and placeholders for label and index
            return None, -1, -1 # Return placeholders that can be skipped in the loader

        if self.transform:
            image = self.transform(image)

        return image, label, original_index # Return image, 0-based label, and original index

# --- Custom Collate Function to handle None ---
def collate_fn(batch):
    # Filter out None entries (where image is None)
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        # Return tensors with appropriate dimensions but empty
        # Assuming image shape will be (3, 224, 224) after transform, adjust if needed
        dummy_image_shape = (3, 224, 224)
        return torch.empty(0, *dummy_image_shape), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)


    images, labels, original_indices = zip(*batch)

    # Stack images and labels
    images = torch.stack(images, 0)
    labels = torch.tensor(labels, dtype=torch.long) # Classification labels are long
    original_indices = torch.tensor(original_indices, dtype=torch.long)

    return images, labels, original_indices


# --- Data Transformations ---
input_size = 224
num_classes = 15

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Mild augmentation
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


# --- DDP-Aware Training Function ---
def train_model_ddp(rank, world_size, model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs, best_model_save_path, label_to_region_id_map, val_set_expected_size):
    since = time.time()

    best_model_wts_state = None # Store state_dict directly
    best_acc = 0.0 # Track best accuracy
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        epoch_str = f'Epoch {epoch+1}/{num_epochs}'
        if rank == 0: # Only print epoch headers on rank 0
            print(epoch_str)
            print('-' * len(epoch_str))

        # --- Set epoch for DistributedSampler - crucial for shuffling! ---
        if world_size > 1:
            dataloaders['train'].sampler.set_epoch(epoch)
            # dataloaders['val'].sampler.set_epoch(epoch) # Not strictly needed for validation if not shuffling, but good practice

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples_this_rank = 0 # Track samples processed in this phase on this rank

            # --- Use tqdm only on rank 0 ---
            disable_tqdm = rank != 0
            data_iterator = tqdm(dataloaders[phase], desc=f"Rank {rank} {phase.capitalize()}", leave=False, file=sys.stdout, disable=disable_tqdm)

            for inputs, labels, _ in data_iterator: # Dataloader yields inputs, labels, original_indices
                if inputs is None or labels is None or inputs.size(0) == 0:
                    if rank == 0: print(f"Rank {rank}: Skipping empty batch in {phase} phase.")
                    continue # Skip empty batches (can happen with collate_fn and small last batches)
                inputs = inputs.to(device)
                labels = labels.to(device)  # Labels are 0-based indices

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # Predicted 0-based indices
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples_this_rank += inputs.size(0)

                # Update tqdm postfix only on rank 0 based on its own batch stats
                if rank == 0:
                    data_iterator.set_postfix(loss=loss.item(),
                                              accuracy=torch.sum(preds == labels.data).item() / inputs.size(0))

            # --- Synchronize Metrics Across All Ranks ---
            # Sum loss, corrects, and samples across all ranks
            # Create tensors on the correct device before all_reduce
            epoch_loss_tensor = torch.tensor([running_loss], device=device)
            if world_size > 1:
                dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)

            running_corrects_tensor = torch.tensor([running_corrects], device=device)
            if world_size > 1:
                dist.all_reduce(running_corrects_tensor, op=dist.ReduceOp.SUM)

            total_samples_tensor = torch.tensor([total_samples_this_rank], device=device)
            if world_size > 1:
                dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)


            # Calculate average metrics across all data
            # Use .item() only *after* aggregation and division
            # Avoid division by zero if total_samples_tensor is 0
            if total_samples_tensor.item() > 0:
                epoch_loss = epoch_loss_tensor.item() / total_samples_tensor.item()
                epoch_acc = running_corrects_tensor.item() / total_samples_tensor.item()
            else:
                 epoch_loss = 0.0
                 epoch_acc = 0.0


            # --- Logging and Saving (Only on Rank 0) ---
            if rank == 0:
                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc)
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Save best model based on validation accuracy (on rank 0)
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # --- Save model state_dict (access via .module with DDP if used) ---
                    if world_size > 1:
                        best_model_wts_state = copy.deepcopy(model.module.state_dict())
                    else:
                        best_model_wts_state = copy.deepcopy(model.state_dict())

                    torch.save(best_model_wts_state, best_model_save_path)
                    print(f"Rank 0: Best model weights (Val Acc: {best_acc:.4f}) saved to {best_model_save_path}")

        # Ensure all processes finish the epoch before starting the next
        if world_size > 1:
            dist.barrier()

    time_elapsed = time.time() - since
    if rank == 0:
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

    # Return the best model state (only relevant for rank 0 for saving/submission) and history
    return best_model_wts_state if rank == 0 else None, history if rank == 0 else None


# --- Submission Generation Function (Only runs on Rank 0) ---
def generate_submission(model, val_loader, label_to_region_id_map, device, output_dir, roll_no, version, val_set_actual_size, val_set_expected_size):
    """
    Generates the submission CSV file using a non-distributed validation loader.

    Args:
        model: The trained model (loaded with best weights, on device).
        val_loader: DataLoader for the validation set (NON-DISTRIBUTED).
        label_to_region_id_map: Dictionary mapping 0-based index back to original Region ID.
        device: The device the model is on ('cuda' or 'cpu').
        output_dir: Directory to save the submission file.
        roll_no: Student's roll number.
        version: Submission version number.
        val_set_actual_size: The actual number of samples in the filtered validation dataset.
        val_set_expected_size: The original number of rows in labels_val.csv (used for ID mapping).
    """
    print("\nRank 0: Generating submission file...")
    model.eval()  # Set model to evaluation mode

    # Store predictions mapped to their original index
    # Dictionary format: {original_index: predicted_region_id}
    predictions_map = {}

    # Process validation set using the non-distributed loader
    with torch.no_grad():
        for inputs, _, original_indices in tqdm(val_loader, desc="Rank 0 Predicting on Validation Set", file=sys.stdout):
            if inputs is None or inputs.size(0) == 0: continue # Skip empty batches
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds_indices = torch.max(outputs, 1) # Get predicted 0-based indices

            # Move predictions and indices to CPU and convert
            preds_cpu = preds_indices.cpu().numpy()
            indices_cpu = original_indices.cpu().numpy()

            # Store predictions mapped by their original index
            for i in range(len(inputs)):
                original_idx = indices_cpu[i]
                predicted_label = preds_cpu[i]
                predictions_map[original_idx] = label_to_region_id_map[predicted_label]

    # --- Create DataFrame for submission ---
    num_val_predictions = len(predictions_map)
    if num_val_predictions != val_set_actual_size:
        print(f"Rank 0: WARNING: Number of stored predictions ({num_val_predictions}) doesn't match filtered validation set size ({val_set_actual_size}). There might be an issue.")

    # Create submission rows based on the original order (0 to N-1, where N is val_set_expected_size)
    submission_data = []
    id_counter = 0
    # --- Use val_set_expected_size for iteration to maintain original ID structure ---
    # This handles cases where some validation images might be missing in the filtered dataset
    for original_idx in range(val_set_expected_size):
        if original_idx in predictions_map:
            region_id = predictions_map[original_idx]
            submission_data.append({'id': id_counter, 'Region_ID': region_id})
            id_counter += 1
        else:
            # Handle missing validation samples as per competition rules if needed
            # For now, we assume we only submit for images that were successfully processed.
            # If the spec requires a row for EVERY original validation entry, you'd need to
            # add a placeholder row here (e.g., Region_ID = 1 or some default).
            pass # Skip if image for this original_idx was missing/skipped

    # Check if the number of generated rows matches the count of actual predictions made
    if id_counter != num_val_predictions:
       print(f"Rank 0: ERROR: Mismatch between generated submission rows ({id_counter}) and number of predictions ({num_val_predictions}). Check ID mapping logic.")


    df_val_preds = pd.DataFrame(submission_data)

    # Test placeholder part
    # The total samples in the final submission should be 738.
    # The remaining IDs after validation predictions are for the test set placeholders.
    num_test_placeholders = 738 - len(df_val_preds) # Calculate based on how many validation rows were included
    start_test_id = len(df_val_preds) # Test IDs start after the last validation ID
    test_ids = list(range(start_test_id, 738)) # Range should go up to 738
    test_region_ids = [1] * num_test_placeholders # Fill with Region_ID = 1


    df_test_placeholders = pd.DataFrame({
        'id': test_ids,
        'Region_ID': test_region_ids
    })

    # Combine validation predictions and test placeholders
    df_submission = pd.concat([df_val_preds, df_test_placeholders], ignore_index=True)

    # Final check on total rows expected: 738 data rows
    if len(df_submission) != 738:
        print(f"Rank 0: ERROR: Final submission DataFrame has {len(df_submission)} data rows, expected 738.")
    else:
        print(f"Rank 0: Submission DataFrame created with {len(df_submission)} data rows.")


    # Save the submission file
    submission_filename = f"{roll_no}_{version}.csv"
    submission_path = os.path.join(output_dir, submission_filename)
    # Ensure columns are exactly 'id', 'Region_ID' in that order
    df_submission[['id', 'Region_ID']].to_csv(submission_path, index=False)
    print(f"Rank 0: Submission file saved to: {submission_path}")



# --- Setup DDP (Initialization is now here and unconditional) ---
# Get Rank and World Size from Environment (set by torchrun/srun)
# Moved here to be accessible before setup
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1)) # Defaults for single process case

setup(rank, world_size)
# Ensure all processes are set up before proceeding
dist.barrier()

torch.cuda.empty_cache() # Clear any cached memory after setup

# Check if data paths exist (only on rank 0, but ensure all ranks wait)
if rank == 0:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(TRAIN_IMG_DIR) or not os.path.isdir(TRAIN_IMG_DIR):
        print(f"ERROR: Training image directory not found or not a directory: {TRAIN_IMG_DIR}")
    if not os.path.exists(VAL_IMG_DIR) or not os.path.isdir(VAL_IMG_DIR):
        print(f"ERROR: Validation image directory not found or not a directory: {VAL_IMG_DIR}")
    if not os.path.exists(TRAIN_CSV):
        print(f"ERROR: Training CSV not found: {TRAIN_CSV}")
    if not os.path.exists(VAL_CSV):
        print(f"ERROR: Validation CSV not found: {VAL_CSV}")
dist.barrier() # Wait for rank 0 to potentially print errors/create dirs


# --- Determine num_classes and label mappings ---
# This needs to be done on all ranks to ensure consistent mappings
num_classes = -1 # Initialize
region_id_to_label = None
label_to_region_id = None
VAL_SET_EXPECTED_SIZE = 362 # Default based on previous code

try:
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV) # Load val CSV to check size later

    # List of filenames with abnormal locations to remove (assuming same cleaning as original)
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
    df_train = df_train[~df_train['filename'].isin(abnormal_filenames)].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Ensure Region_ID is integer
    df_train['Region_ID'] = df_train['Region_ID'].astype(int)

    # Get unique, sorted original Region_IDs
    unique_region_ids = sorted(df_train['Region_ID'].unique())
    num_classes = len(unique_region_ids)

    # --- Crucial Check: Ensure Region IDs are within the expected range (1-15) ---
    valid_ids = all(1 <= rid <= 15 for rid in unique_region_ids)
    if not valid_ids:
        raise ValueError(f"Error: Region IDs found outside the expected range [1, 15]. Found: {unique_region_ids}")

    # Create mapping from original Region_ID (1-15) to 0-based label (0 to num_classes-1)
    region_id_to_label = {region_id: i for i, region_id in enumerate(unique_region_ids)}
    # Create reverse mapping from 0-based label back to original Region_ID
    label_to_region_id = {i: region_id for region_id, i in region_id_to_label.items()}

    if rank == 0:
        print(f"Found {num_classes} unique Region_IDs in training data: {unique_region_ids}")
        print(f"Mapping Region_ID to internal 0-based labels: {region_id_to_label}")
        print(f"Mapping internal 0-based labels back to Region_ID: {label_to_region_id}")

    # --- Check Validation Set Size ---
    VAL_SET_EXPECTED_SIZE = 362
    if rank == 0:
        if len(df_val) != VAL_SET_EXPECTED_SIZE:
            print(f"WARNING: Validation set size ({len(df_val)}) does not match expected size ({VAL_SET_EXPECTED_SIZE}). Submission generation might be incorrect.")
        else:
            print(f"Validation set size confirmed: {len(df_val)}")

except Exception as e:
    if rank == 0:
        print(f"Error reading CSVs or processing Region_ID: {e}")
    num_classes = -1
    region_id_to_label = None
    label_to_region_id = None


# --- Create Datasets and DataLoaders ---
dataloaders = None # Initialize to None
VAL_SET_ACTUAL_SIZE = 0 # Initialize actual size

if num_classes > 0 and region_id_to_label:
    try:
        if rank == 0: print("\nCreating Datasets...")
        image_datasets = {
            'train': CustomImageDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=data_transforms['train'], label_map=region_id_to_label),
            'val': CustomImageDataset(VAL_CSV, VAL_IMG_DIR, transform=data_transforms['val'], label_map=region_id_to_label) # Use same map for val
        }

        # Check if datasets were loaded successfully and are not empty
        if len(image_datasets['train']) == 0 or len(image_datasets['val']) == 0:
             raise ValueError("One or both datasets are empty after filtering. Check image paths and CSV contents.")

        # --- Adjust validation dataset size check after filtering ---
        VAL_SET_ACTUAL_SIZE = len(image_datasets['val'])
        if rank == 0:
            print(f"Actual validation dataset size after filtering: {VAL_SET_ACTUAL_SIZE}")
            if VAL_SET_ACTUAL_SIZE != VAL_SET_EXPECTED_SIZE:
                 print(f"WARNING: Actual validation set size ({VAL_SET_ACTUAL_SIZE}) does not match expected size ({VAL_SET_EXPECTED_SIZE}) after filtering. Submission IDs might need adjustment if files were missing.")


        if rank == 0: print("\nCreating DataLoaders with Distributed Samplers...")
        # +++ Create Distributed Samplers +++
        # Only create samplers if world_size > 1, otherwise DataLoader handles shuffling
        if world_size > 1:
            train_sampler = DistributedSampler(image_datasets['train'], num_replicas=world_size, rank=rank, shuffle=True) # Shuffle=True for training
            val_sampler = DistributedSampler(image_datasets['val'], num_replicas=world_size, rank=rank, shuffle=False) # Shuffle=False for validation
            train_shuffle = False
            val_shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            train_shuffle = True # Shuffle in DataLoader for single process
            val_shuffle = False


        # +++ Create DataLoaders with Samplers (or without for single process) +++
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=train_shuffle, num_workers=NUM_WORKERS, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn),
            'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=val_shuffle, num_workers=NUM_WORKERS, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)
        }
        # Dataset sizes per rank (approximate for reporting)
        # Use the actual size of the dataset (after filtering) for accurate total
        dataset_sizes = {
            'train': len(image_datasets['train']),
            'val': len(image_datasets['val'])
        }

        if rank == 0:
             print(f"Training dataset total size: {dataset_sizes['train']}")
             print(f"Validation dataset total size: {dataset_sizes['val']}")
             print(f"Effective Total Batch Size (across all GPUs or CPU): {BATCH_SIZE * world_size}")

    except Exception as e:
        if rank == 0: print(f"Rank {rank}: Error creating Datasets or DataLoaders: {e}")
        dataloaders = None
else:
    if rank == 0: print("Cannot proceed without determining the number of classes and label map.")
    dataloaders = None

# --- Model Setup ---
model = None # Initialize model to None
criterion = None # Initialize criterion to None
optimizer = None # Initialize optimizer to None
device = None # Initialize device

if num_classes > 0 and dataloaders:
    # Determine device based on rank (and if cuda is available)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0: print(f"Rank {rank}: Using device: {device}")

    weights = ResNet18_Weights.IMAGENET1K_V1 # Or IMAGENET1K_V2 for better accuracy
    # Use default weights for all ranks, load state dict later if needed from rank 0 checkpoint
    model = models.resnet18(weights=weights)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # Replace the final layer
    if rank == 0: print(f"Rank {rank}: Model final layer adapted for {num_classes} classification classes.")

    # +++ Move model to the correct device for this rank BEFORE wrapping +++
    model = model.to(device)
    if rank == 0: print(f"Rank {rank}: Model moved to device: {device}")

    # +++ Wrap model with DDP +++
    # Only wrap with DDP if world_size > 1
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        if rank == 0: print(f"Rank {rank}: Model wrapped with DDP across {world_size} GPUs.")
    else:
        if rank == 0: print("Running in single-process mode, DDP not applied.")


    # --- Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    if rank == 0: print(f"Rank {rank}: Using Loss Function: {criterion.__class__.__name__}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if rank == 0: print(f"Rank {rank}: Using Optimizer: Adam with LR={LEARNING_RATE}")


# --- Start Training ---
if dataloaders and model and criterion and optimizer and label_to_region_id and device:
    if rank == 0: print("\nStarting DDP classification training...")

    best_model_state, history = train_model_ddp(
        rank, world_size, model, criterion, optimizer, dataloaders, dataset_sizes,
        device, NUM_EPOCHS, BEST_MODEL_PATH, label_to_region_id, VAL_SET_EXPECTED_SIZE
    ) # Pass rank, world_size, device, paths, mappings, and expected val size

    # --- Generate Submission (Only on Rank 0) ---
    # Check if rank is 0 and if training completed successfully (best_model_state is not None)
    if rank == 0 and best_model_state is not None:
        print("\nRank 0: Loading best model for submission generation...")
        # Re-create the base model structure first on rank 0's device
        # Ensure num_classes is available here
        if num_classes <= 0:
             print("Rank 0: Cannot generate submission, num_classes not determined.")
        else:
            base_model = models.resnet18(weights=None)  # No pretrained weights needed now
            num_ftrs = base_model.fc.in_features
            base_model.fc = nn.Linear(num_ftrs, num_classes)

            # Load the saved state dict (saved by rank 0)
            try:
                # Load to rank 0's device
                base_model.load_state_dict(best_model_state)
                base_model = base_model.to(device)
                if rank == 0: print("Rank 0: Best model state loaded successfully.")

                # Need a non-distributed dataloader for final validation prediction on rank 0
                # This ensures we process the validation set in its original order
                # Ensure label_map is passed to the dataset
                val_dataset_final = CustomImageDataset(VAL_CSV, VAL_IMG_DIR, transform=data_transforms['val'], label_map=region_id_to_label)
                val_loader_final = DataLoader(val_dataset_final, batch_size=BATCH_SIZE*2, # Can use larger batch for inference
                                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn) # Use collate_fn


                generate_submission( # Original function should work if model is on correct device
                    model=base_model,
                    val_loader=val_loader_final, # Use the non-distributed loader
                    label_to_region_id_map=label_to_region_id, # Ensure this is available
                    device=device, # Rank 0's device
                    output_dir=OUTPUT_DIR,
                    roll_no=YOUR_ROLL_NO,
                    version=SUBMISSION_VERSION,
                    val_set_actual_size=len(val_dataset_final), # Use size from the final loader
                    val_set_expected_size=VAL_SET_EXPECTED_SIZE # Ensure this is available
                )

                # Optional Plotting (only on rank 0)
                if history is not None:
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(12, 5))
                        plt.subplot(1, 2, 1)
                        plt.plot(history['train_loss'], label='Train Loss')
                        plt.plot(history['val_loss'], label='Val Loss')
                        plt.xlabel("Epochs")
                        plt.ylabel("Loss")
                        plt.legend()
                        plt.title('Loss vs. Epochs')

                        plt.subplot(1, 2, 2)
                        plt.plot(history['train_acc'], label='Train Accuracy')
                        plt.plot(history['val_acc'], label='Val Accuracy')
                        plt.xlabel("Epochs")
                        plt.ylabel("Accuracy")
                        plt.legend()
                        plt.title('Accuracy vs. Epochs')
                        plt.tight_layout()
                        plot_path = os.path.join(OUTPUT_DIR, f'training_plot_{YOUR_ROLL_NO}_v{SUBMISSION_VERSION}.png')
                        plt.savefig(plot_path)
                        print(f"Rank 0: Training plot saved to {plot_path}")
                        # plt.show() # Comment out if running non-interactively
                    except ImportError:
                        print("Rank 0: Matplotlib not found. Skipping plotting.")
                    except Exception as e:
                        print(f"Rank 0: An error occurred during plotting: {e}")
                else:
                     print("Rank 0: Training history not available for plotting.")


            except FileNotFoundError:
                 print(f"Rank 0: ERROR: Best model checkpoint not found at {BEST_MODEL_PATH}. Cannot generate submission.")
            except Exception as e:
                print(f"Rank 0: An error occurred during submission generation: {e}")


    elif rank == 0: # Training prerequisites met, but best_model_state is None
        print("Rank 0: Training did not produce a best model state. Submission not generated.")

else:
    if rank == 0: print("\nTraining prerequisites not met (check data paths, CSV files, or initial setup steps).")
    if rank == 0: print("Submission file not generated.")

# --- Clean up DDP ---
# Cleanup should be called regardless of world_size if init_process_group was called
cleanup()

if rank == 0: print("Main script finished.")