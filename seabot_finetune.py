# Standard library imports
import glob
import os
import random
import re
import requests
import traceback
import time
import timm

# External libraries for data handling and analysis
import numpy as np
import pandas as pd
from collections import Counter

# Imaging and visualization libraries
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning and machine learning libraries
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# FathomNet API for marine biodiversity data
from fathomnet.api import images, boundingboxes, taxa

# Utilities and tools
from tqdm import tqdm

# Cloud and external services
import boto3
import wandb

# Import Dynaconf and load configurations
from dynaconf import Dynaconf

# Initialize settings
settings = Dynaconf(settings_files=['setting.toml'])

# Replace hardcoded values with configuration values
BUCKET_NAME = settings.BUCKET_NAME
S3_MODEL_ROOT_PATH = settings.S3_MODEL_ROOT_PATH
WANDB_KEY = settings.WANDB_KEY
MODEL_ROOT_PATH = settings.MODEL_ROOT_PATH
MODEL_NAME = settings.MODEL_NAME
OLD_MODEL_PATH = settings.OLD_MODEL_PATH
FATHOMNET_RELATIVE_PATH = settings.FATHOMNET_RELATIVE_PATH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Generate a dynamic run name for wandb based on the model name
run_name = f"fn_finetuning_{MODEL_NAME}_pretrained"
wandb.login(key=WANDB_KEY)
wandb.init(project="seabot", name=run_name)

def download_from_s3(bucket_name, s3_path, local_path, model_name):
    local_model_path = os.path.join(local_path, f'{model_name}.pth')
    try:
        s3_client.download_file(bucket_name, s3_path, local_model_path)
        print(f"Successfully downloaded {s3_path} to {local_model_path}")
    except Exception as e:
        print(f"Error occurred while downloading from S3: {e}")

def save_model_to_s3(local_model_path, s3_path, bucket_name, model_name):
    s3_model_path = os.path.join(s3_path, f'{model_name}.pth')
    try:
        s3_client.upload_file(local_model_path, bucket_name, s3_model_path)
        print(f"Model successfully uploaded to {s3_model_path}")
    except Exception as e:
        print(f"Error occurred while uploading model to S3: {e}")

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
        max_retries = 3  # Maximum number of retries for downloading an image
        retry_delay = 2  # Delay in seconds between retries

        for image_info in tqdm(self.images_info, desc="Downloading images", unit="image"):
            image_url = image_info.url
            image_path = os.path.join(self.image_dir, f"{image_info.uuid}.jpg")

            if not os.path.exists(image_path):
                for attempt in range(max_retries):
                    try:
                        image_data = requests.get(image_url).content
                        with open(image_path, 'wb') as handler:
                            handler.write(image_data)
                        break  # Break out of the loop if download is successful
                    except Exception as e:
                        print(f"Attempt {attempt + 1} failed for {image_url}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)  # Wait before retrying
                        else:
                            print(f"Failed to download image after {max_retries} attempts.")
                else:
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

    # Load the pre-trained Vision Transformer model and replace the classifier with a new one with 4 classes
    model = timm.create_model(MODEL_NAME, pretrained=True)

    # Replace the classifier with a new one tailored to the number of classes
    num_classes = len(concepts)
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):  # For models with 'fc' layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Unfreeze all layers for training
    for param in model.parameters():
        param.requires_grad = True
    
    # Define the model names
    old_model_name = settings.OLD_MODEL_NAME  # Assuming OLD_MODEL_NAME is defined in settings
    model_name = settings.MODEL_NAME
    
    # Load the pre-trained model parameters for further training
    model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
    
    # Download and load the pre-trained model parameters for further training
    if old_model_path:
        local_old_model_path = os.path.join(model_root_path, f'{old_model_name}.pth')
        s3_old_model_path = os.path.join(S3_MODEL_ROOT_PATH, f'{old_model_name}.pth')
        download_from_s3(BUCKET_NAME, s3_old_model_path, model_root_path, old_model_name)
    
        if os.path.isfile(local_old_model_path):
            model.load_state_dict(torch.load(local_old_model_path))
            print(f"Loaded the pre-trained model from {local_old_model_path} for further training.")
        else:
            print(f"Pre-trained model file {local_old_model_path} not found. Using default weights.")


    # Replace the classifier again, this time with the number of concept classes
    model.classifier = nn.Linear(model.config.hidden_size, len(concepts))

    # Move the model to the GPU if available
    model = model.to(device)

    # Define the optimizer as Adam
    optimizer = optim.Adam(model.parameters())

    # Define the number of training epochs and the patience for early stopping
    num_epochs = 1
    patience = 2
    no_improve_epoch = 0

    # Frequency for saving the model
    save_freq = 100000

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
        total_batches = len(train_loader)  # Total number of batches in one epoch
        for epoch in range(start_epoch, num_epochs):
            print(f'Starting epoch {epoch + 1}/{num_epochs}')
            running_loss = 0.0
            model.train()

            for batch_idx, (images, labels_vector) in enumerate(train_loader, start=start_batch):
                if images is None or labels_vector is None:
                    print("Terminating batch due to image or label vector read error.")
                    break

                images = images.to(device)
                labels_vector = labels_vector.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs.logits, labels_vector)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                wandb.log({"fn_epoch": epoch, "fn_loss": loss.item()})

                # Print epoch progress
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Epoch {epoch + 1} Progress: {progress:.2f}%")

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
            
                # Save the best model to S3
                best_model_name = f'{settings.MODEL_NAME}_best'
                best_model_path = os.path.join(checkpoint_folder, f'{best_model_name}.pth')
                torch.save(model.state_dict(), best_model_path)
                s3_model_path = os.path.join(S3_MODEL_ROOT_PATH, f'{best_model_name}.pth')
                save_model_to_s3(best_model_path, s3_model_path, BUCKET_NAME, best_model_name)
            else:
                no_improve_epoch += 1

            if no_improve_epoch >= patience:
                print(f'Early stopping after {patience} epochs without improvement.')
                break

    # Load the latest checkpoint and start/resume training
    start_epoch, start_batch, best_loss = load_latest_checkpoint()
    train_loop(start_epoch, start_batch, best_loss)

    final_model_name = f'{settings.MODEL_NAME}_final'
    final_model_path = os.path.join(checkpoint_folder, f'{final_model_name}.pth')
    torch.save(model.state_dict(), final_model_path)
    s3_final_model_path = os.path.join(S3_MODEL_ROOT_PATH, f'{final_model_name}.pth')
    save_model_to_s3(final_model_path, s3_final_model_path, BUCKET_NAME, final_model_name)

# Use the configuration values for paths
model_root_path = MODEL_ROOT_PATH
old_model_path = OLD_MODEL_PATH
current_working_dir = os.getcwd()
fathomnet_root_path = os.path.join(current_working_dir, FATHOMNET_RELATIVE_PATH)

load_and_train_model(model_root_path, old_model_path, fathomnet_root_path)
