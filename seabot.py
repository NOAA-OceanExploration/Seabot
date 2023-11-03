# -*- coding: utf-8 -*-
"""

This version of seabot is intended for training and deployment via a EC2 instance.

Original file is located at
    https://colab.research.google.com/drive/1Gcth1dGuMimPLkRt3jvYn7MTNUUg4rf0

# SeaBot - Image

Train both the image model and the text generation model at the same time. Such that the embeddings of the video classification model are passed directly into the text generation model. Vanishing gradient could be a problem, I may have to do some skipped connections, make it a bit more shallow...

Open Question: Is the self-supervised fine-tuning process necessary?

Structure:

1. Train a generator to create instance of the distribution of annotation text. Unsupervised.
2. Train the video classification method on the distribution of the annotation imagery. Unsupervised.
3. Combine both methods into a singular pipeline and then use the actual annotations to derive results.

The Models:


The Data:

## Setup
"""

from google.colab import drive
drive.mount('/content/drive')

"""## Install Necessary Dependencies"""

!pip install transformers
!pip install torch
!pip install pytorchvideo
!pip install ffmpeg-python
!pip install torchvision
!pip install tqdm
!pip install fathomnet
!pip install openai
!pip install wandb
!pip install pafy

"""## Fine-tuning Image Vision Model (videomae-base)"""

from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from random import randint
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from fathomnet.api import images, boundingboxes, taxa
from tqdm import tqdm
from collections import Counter

import ffmpeg
import glob
import random
import numpy as np
import openai
import os
import re
import requests
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

"""### Fathomnet Fine-Tuning"""

# Define the custom dataset class for handling FathomNet data
class FathomNetDataset(Dataset):
    def __init__(self, fathomnet_root_path, concepts, transform=None):
        self.transform = transform
        self.images_info = []
        self.image_dir = fathomnet_root_path
        self.concepts = concepts
        self.concept_to_index = {concept: i for i, concept in enumerate(concepts)}

        print("Number of classes in set: " + str(len(concepts)))

        # Fetch image data for each concept and save the information
        for concept in concepts:
            try:
                images_info = images.find_by_concept(concept)
                self.images_info.extend(images_info)
            except ValueError as ve:
                print(f"Error fetching image data for concept {concept}: {ve}")
                continue

        # Sort images info to ensure consistent order across different runs
        self.images_info.sort(key=lambda x: x.uuid)

        # Create directory if it doesn't exist
        os.makedirs(self.image_dir, exist_ok=True)

        # Download images for each image info and save it to disk
        for image_info in tqdm(self.images_info, desc="Downloading images", unit="image"):
          image_url = image_info.url
          image_path = os.path.join(self.image_dir, f"{image_info.uuid}.jpg")

          # Download only if image doesn't already exist
          if not os.path.exists(image_path):
              try:
                  image_data = requests.get(image_url).content
                  with open(image_path, 'wb') as handler:
                      handler.write(image_data)
              except ValueError as ve:
                  print(f"Error downloading image from {image_url}: {ve}")
                  continue

    # Get the number of images in the dataset
    def __len__(self):
        return len(self.images_info)

    # Fetch an image and its label vector by index
    def __getitem__(self, idx):
      try:
          image_info = self.images_info[idx]
          image_path = os.path.join(self.image_dir, f"{image_info.uuid}.jpg")
          image = Image.open(image_path).convert('RGB')

          # Create label vector
          labels_vector = torch.zeros(len(self.concepts))
          for box in image_info.boundingBoxes:
            if box.concept in self.concept_to_index:
              labels_vector[self.concept_to_index[box.concept]] = 1

          # Apply transformations if any
          if self.transform:
            image = self.transform(image)

          return image, labels_vector
      except (IOError, OSError):
          print(f"Error reading image {image_path}. Skipping.")
          return None, None

def collate_fn(batch):
    # Filter out the (None, None) entries from the batch
    batch = [(image, label) for image, label in batch if image is not None and label is not None]

    # If there are no valid items left, return (None, None)
    if len(batch) == 0:
        return None, None

    # Extract and stack the images and labels
    images = torch.stack([item[0] for item in batch])
    labels_vector = torch.stack([item[1] for item in batch])

    return images, labels_vector

def load_and_train_model(model_root_path, old_model_path, fathomnet_root_path):
    # Define a transformation that resizes images to 224x224 pixels and then converts them to tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Find the concepts for the bounding boxes
    concepts = boundingboxes.find_concepts()

    # Create a dataset with the given concepts and the defined transform
    dataset = FathomNetDataset(fathomnet_root_path, concepts, transform=transform)

    # Calculate the sizes for the training and validation datasets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Set a seed for the random number generator
    torch.manual_seed(0)

    # Split the dataset into training and validation subsets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders for the training and validation datasets with batch size of 16
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

    # Load the pre-trained Vision Transformer model and replace the classifier with a new one with 4 classes
    model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
    model.classifier = nn.Linear(model.config.hidden_size, 4)

    # Unfreeze all layers for training
    for param in model.parameters():
        param.requires_grad = True

    # Load the pre-trained model parameters for further training
    model.load_state_dict(torch.load(old_model_path))
    print("Loaded the d2 model parameters for further training")

    # Replace the classifier again, this time with the number of concept classes
    model.classifier = nn.Linear(model.config.hidden_size, len(concepts))

    # Move the model to the GPU if available
    model = model.to(device)

    # Define the optimizer as Adam
    optimizer = optim.Adam(model.parameters())

    # Define the number of training epochs and the patience for early stopping
    num_epochs = 5
    patience = 2
    no_improve_epoch = 0

    # Frequency for saving the model
    save_freq = 100

    # Replace the StepLR with OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=num_epochs)

    # Define a folder to store checkpoints
    checkpoint_folder = os.path.join(model_root_path, 'fn_checkpoints')

    # Make sure the checkpoint folder exists
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Load the latest checkpoint if it exists
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_folder, "*.pth")))

    # Define a function to load the latest checkpoint
    def load_latest_checkpoint():
        checkpoints = glob.glob(os.path.join(checkpoint_folder, "*.pth"))
        checkpoints.sort(key=lambda x: [int(num) for num in re.findall(r'\d+', x)], reverse=True) # Sorting based on epoch and batch number

        if checkpoints:
            latest_checkpoint_path = checkpoints[0]
            checkpoint = torch.load(latest_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch']
            best_loss = checkpoint['best_loss']
            print(f"Loaded Checkpoint from {latest_checkpoint_path}!!")
            return start_epoch, start_batch, best_loss
        else:
            print("No Checkpoint found!!")
            return 0, 0, np.inf


    # Define the loss function as binary cross-entropy with logits
    criterion = nn.BCEWithLogitsLoss()

    # Ensure the model is in the correct device
    model.to(device)

    # Define a function for the training loop
    def train_loop(start_epoch, start_batch, best_loss):
        for epoch in range(start_epoch, num_epochs):
            print(f'Starting epoch {epoch + 1}/{num_epochs + 1}')
            running_loss = 0.0
            model.train()

            for batch_idx, (images, labels_vector) in enumerate(train_loader, start=start_batch):
                if images is None or labels_vector is None:
                    print("Skipping batch due to image or label vector read error.")
                    continue

                images = images.to(device)
                labels_vector = labels_vector.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs.logits, labels_vector)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                wandb.log({"fn_epoch": epoch, "fn_loss": loss.item()})

                if (batch_idx + 1) % save_freq == 0:
                    checkpoint_path = os.path.join(checkpoint_folder, f'fn_checkpoint_{epoch + 1}_{batch_idx + 1}.pth')
                    torch.save({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'best_loss': best_loss,
                    }, checkpoint_path)
                    print(f'Saved model checkpoint at {checkpoint_path}')

            epoch_loss = running_loss / len(train_loader.dataset)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                print(f'New best loss: {best_loss}')
                no_improve_epoch = 0  # Reset patience
            else:
                no_improve_epoch += 1

            if no_improve_epoch >= patience:
                print(f'Early stopping after {patience} epochs without improvement.')
                break

    # Load the latest checkpoint and start/resume training
    start_epoch, start_batch, best_loss = load_latest_checkpoint()
    train_loop(start_epoch, start_batch, best_loss)


# Set the fathomnet_root_path and model_root_path as parameters to be passed to the function
old_model_path = os.path.join(model_root_path, 'd2_fine_tuned_model.pt')
load_and_train_model(model_root_path, old_model_path, fathomnet_root_path)
torch.save(model.state_dict(), os.path.join(model_root_path, 'fn_trained_model.pth'))

"""## Classify and Humanize Outputs

Take the results of this classification model and 'humanize' them.
"""

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def extract_frames(video_path, frame_rate=1):
    # use ffmpeg to extract frames from the video
    pass  # Fill this in

def classify_frames(video_path, model, frame_rate=1):
    # Extract frames from the video
    extract_frames(video_path, frame_rate)

    # Load frames
    frame_dir = video_path.rsplit('.', 1)[0]
    frame_files = sorted(glob.glob(frame_dir + '/*.png'))
    frame_dataset = ImageDataset(frame_files, transform)
    frame_loader = DataLoader(frame_dataset, batch_size=1)  # Classify one frame at a time

    # Classify frames
    model.eval()
    seen_classes = set()
    first_spotted = {}
    with torch.no_grad():
        for i, frame in enumerate(frame_loader):
            frame = frame.to(device)
            output = model(frame.unsqueeze(0))
            _, predicted = torch.max(output.logits.data, 1)
            predicted_class = predicted.item()

            if predicted_class not in seen_classes:
                seen_classes.add(predicted_class)
                timecode = i / frame_rate  # Calculate timecode
                first_spotted[concepts[predicted_class]] = timecode  # concepts is a list of all the concepts

    return first_spotted
