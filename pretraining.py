"""
Description:
    This script is designed to train a Vision Transformer (ViT) model on a dataset of images
    extracted from videos for a project named SeaBot. The images are pre-processed and loaded
    using PyTorch's Dataset and DataLoader classes. The training loop is designed with early
    stopping to prevent overfitting. The script incorporates a checkpointing system to save 
    the model, optimizer, and scheduler states, allowing for resumption of training from the 
    last checkpoint in case of interruption. The Weights & Biases (wandb) library is used for
    logging the training and validation loss. Additionally, utility functions are provided for
    tasks like checking image integrity, extracting frames from videos, and loading the latest 
    checkpoint. The script is configured to run on GPU if available, otherwise on CPU.

Dependencies:
    - torch
    - torchvision
    - transformers
    - sklearn
    - PIL
    - ffmpeg
    - wandb
    - glob, os, random, numpy, requests, re, traceback, tqdm, concurrent.futures
"""

from PIL import Image
from random import randint
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from tqdm import tqdm

import ffmpeg
import glob
import random
import numpy as np
import os
import torch
import wandb
import traceback

# Setup for using GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set random seeds for reproducibility

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Initialize a new run
wandb.init(project="seabot", config={
    "architecture": "vit-base-patch16-224",
    "dataset": "fathomnet & d2_dives",
})

dataset_root_path = "/content/drive/MyDrive/Colab Notebooks/Work/SeaBot/Data"
fathomnet_root_path = "drive/MyDrive/Colab Notebooks/Work/SeaBot/FathomNet"
model_root_path = "drive/MyDrive/Colab Notebooks/Work/SeaBot"

if os.path.exists(dataset_root_path):
  print('The directory exists')
else:
  print('The directory does not exist')

"""### Dive Pretraining"""

from PIL import Image
import glob
import os

def check_image_integrity(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Corrupted file {image_path}: {e}")
        return False

# Define paths
checkpoint_folder = os.path.join(model_root_path, 'd2_checkpoints')
model_path = os.path.join(model_root_path, 'd2_fine_tuned_model.pt')

# Define hyperparameters
num_epochs = 2
patience = 2
save_freq = 1000

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize images
    transforms.ToTensor(),  # convert to tensor
])

# Define model
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
model.classifier = nn.Linear(model.config.hidden_size, 4)
model = model.to(device)

# Define optimizer and scheduler
optimizer = optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Create checkpoint folder if it doesn't exist
os.makedirs(checkpoint_folder, exist_ok=True)

# Define Dataset class
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        rotation = randint(0, 3)  # Randomly choose a rotation
        rotated_image = image.rotate(rotation * 90)
        if self.transform:
            rotated_image = self.transform(rotated_image)
        return rotated_image, rotation

# Define function to extract frames from video
def extract_frames(video_path, frame_rate=1):
    out_path = video_path.rsplit('.', 1)[0]  # Remove file extension
    if os.path.exists(out_path):  # If frames have already been extracted, skip
        return

    os.makedirs(out_path, exist_ok=True)  # Create directory to save frames

    (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=frame_rate)
        .output(os.path.join(out_path, 'img%03d.png'))
        .run()
    )

# Define function to load the latest checkpoint
def load_latest_checkpoint(model, optimizer, scheduler):
    try:
        checkpoints = glob.glob(os.path.join(checkpoint_folder, "*.pth"))
        checkpoints.sort(key=os.path.getmtime) # Sorting based on last modification time

        if checkpoints:
            latest_checkpoint_path = checkpoints[-1]
            print(f"Attempting to load checkpoint from {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch']
            best_loss = checkpoint['best_loss']
            print(f"Loaded Checkpoint from {latest_checkpoint_path}!!")
            return model, optimizer, scheduler, start_epoch, start_batch, best_loss
        else:
            print("No Checkpoint found!!")
            return model, optimizer, scheduler, 0, 0, np.inf
    except Exception as e:
        print(f"Exception occurred: {e}")
        traceback.print_exc()
        raise e

# Define training loop
def train_loop(start_epoch, start_batch, best_loss):
    print("Fine-tuning model")

    wandb.watch(model, log="all")
    no_improve_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        print(f'Starting epoch {epoch + 1}/{num_epochs}')
        epoch_loss = 0.0
        model.train()
        for batch_idx, (batch, targets) in enumerate(train_loader, start=start_batch):
            # Move batch and targets to device
            batch = batch.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs.logits, targets)
            loss.backward()
            optimizer.step()

            # Update epoch loss
            epoch_loss += loss.item()

            wandb.log({"d2_epoch": epoch, "d2_loss": loss.item()})

            # Print status every 100 batches
            if (batch_idx + 1) % save_freq == 0:
                checkpoint_path = os.path.join(checkpoint_folder, f'd2_checkpoint_{epoch + 1}_{batch_idx + 1}.pth')
                torch.save({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

        scheduler.step()

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (batch, targets) in enumerate(val_loader):
                batch = batch.to(device)
                targets = targets.to(device)

                outputs = model(batch)
                loss = criterion(outputs.logits, targets)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        wandb.log({"d2_val_loss": val_loss})
        print(f'Validation Loss: {val_loss}')

        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epoch = 0
            torch.save(model.state_dict(), model_path)
            print(f'New best model saved at epoch {epoch + 1}')
        else:
            no_improve_epoch += 1
            print(f'Patience counter: {no_improve_epoch}')
            if no_improve_epoch >= patience:
                print('Early stopping...')
                return True

        # Print average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed, Average Loss: {avg_loss}')

    return False  # Indicate that training is not finished

# Extract frames from all mp4 files in subfolders within a directory
video_files = glob.glob(dataset_root_path + '/**/*.mp4', recursive=True)

for video_file in tqdm(video_files, total=len(video_files), desc='Extracting frames'):
    extract_frames(video_file)

if os.path.isfile(model_path):
    print("The model has already been trained. Skipping training.")
else:
    image_files = glob.glob(dataset_root_path + '/**/*.png', recursive=True)

    # Add this before starting the training process
    print("Checking image integrity...")
    image_files = glob.glob(dataset_root_path + '/**/*.png', recursive=True)
    valid_files = [f for f in image_files if check_image_integrity(f)]

    if len(valid_files) < len(image_files):
        print(f"Found {len(image_files) - len(valid_files)} corrupted files. Proceeding with {len(valid_files)} valid files.")

    # Existing code for splitting dataset and training
    # Replace 'image_files' with 'valid_files' in the following lines
    num_files = int(len(valid_files))
    selected_files = np.random.choice(valid_files, size=num_files, replace=False).tolist()

    # Splitting the dataset into train and validation sets
    train_files, val_files = train_test_split(selected_files, test_size=0.2, random_state=42)

    # Creating the Datasets
    train_dataset = ImageDataset(train_files, transform)
    val_dataset = ImageDataset(val_files, transform)

    # Creating the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    print("Starting the training process.")
    try:
        model, optimizer, scheduler, start_epoch, start_batch, best_loss = load_latest_checkpoint(model, optimizer, scheduler)
        train_loop(start_epoch, start_batch, best_loss)
    except Exception as e:
        print(f"Error occurred during training: {e}. Exiting...")
