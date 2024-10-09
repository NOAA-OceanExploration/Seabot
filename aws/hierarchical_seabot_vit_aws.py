import boto3
import os
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from fathomnet.api import images, boundingboxes, taxa
from tqdm import tqdm
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ffmpeg
import glob
import random
import numpy as np
import openai
import re
import requests
import torch
import traceback

# AWS S3 setup
s3_bucket = 'your-s3-bucket-name'
s3_data_prefix = 'your/dataset/prefix/'
local_data_dir = '/tmp/dataset'

# Create local directories if they don't exist
os.makedirs(local_data_dir, exist_ok=True)

# Boto3 client for S3 operations
s3 = boto3.client('s3')

# Function to download files from S3
def download_from_s3(bucket, s3_key, local_file):
    s3.download_file(bucket, s3_key, local_file)

# Download entire dataset from S3
def download_s3_dataset(bucket, s3_prefix, local_dir):
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)
    
    for page in pages:
        for obj in page['Contents']:
            s3_key = obj['Key']
            local_file = os.path.join(local_dir, os.path.basename(s3_key))
            if not os.path.exists(local_file):
                print(f"Downloading {s3_key} to {local_file}")
                download_from_s3(bucket, s3_key, local_file)

# Download the dataset
download_s3_dataset(s3_bucket, s3_data_prefix, local_data_dir)

# Setup for GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# WandB setup
import wandb
wandb.login(key='your_wandb_key_here')

wandb.init(
    project="seabot",
    name=f"drive_pretraining_2",
)

# Path setup for dataset and models
checkpoint_folder = os.path.join(local_data_dir, 'd2_checkpoints')
model_path = os.path.join(local_data_dir, 'd2_fine_tuned_model.pt')

# Hyperparameters
num_epochs = 5
patience = 2
save_freq = 100

# Model and transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load pre-trained ViT model
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
model.classifier = nn.Linear(model.config.hidden_size, 4)

# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

model = model.to(device)

# Optimizer, scheduler, and loss function
optimizer = optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Ensure checkpoint folder exists
os.makedirs(checkpoint_folder, exist_ok=True)

# Dataset class for images
class ImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        rotation = randint(0, 3)  # Randomly rotate the image
        rotated_image = image.rotate(rotation * 90)
        if self.transform:
            rotated_image = self.transform(rotated_image)
        return rotated_image, rotation

# Function to extract frames from video
def extract_frames(video_path, frame_rate=1):
    out_path = video_path.rsplit('.', 1)[0]  # Remove file extension
    if os.path.exists(out_path):  # Skip if frames already extracted
        return

    os.makedirs(out_path, exist_ok=True)  # Create directory for frames
    (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=frame_rate)
        .output(os.path.join(out_path, 'img%03d.png'))
        .run()
    )

# Pretraining Setup
# Define Dataset class for pretraining
class PretrainingDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        rotation = randint(0, 3)  # Random rotation for data augmentation
        rotated_image = image.rotate(rotation * 90)
        if self.transform:
            rotated_image = self.transform(rotated_image)
        return rotated_image, rotation

# Pretraining: Load the dataset, define dataloaders
def prepare_pretraining_data(dataset_root_path, num_files):
    image_files = glob.glob(dataset_root_path + '/**/*.png', recursive=True)
    selected_files = np.random.choice(image_files, size=num_files, replace=False).tolist()

    train_files, val_files = train_test_split(selected_files, test_size=0.2, random_state=42)

    train_dataset = PretrainingDataset(train_files, transform=transform)
    val_dataset = PretrainingDataset(val_files, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    return train_loader, val_loader

# Load the latest checkpoint
def load_latest_checkpoint(model, optimizer, scheduler):
    try:
        checkpoints = glob.glob(os.path.join(checkpoint_folder, "*.pth"))
        checkpoints.sort(key=os.path.getmtime)

        if checkpoints:
            latest_checkpoint_path = checkpoints[-1]
            print(f"Loading checkpoint from {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch']
            best_loss = checkpoint['best_loss']
            return model, optimizer, scheduler, start_epoch, start_batch, best_loss
        else:
            return model, optimizer, scheduler, 0, 0, np.inf
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        traceback.print_exc()
        raise e

# Training loop for pretraining
def train_loop(start_epoch, start_batch, best_loss):
    wandb.watch(model, log="all")
    no_improve_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        print(f'Starting epoch {epoch + 1}/{num_epochs}')
        epoch_loss = 0.0
        model.train()

        for batch_idx, (batch, targets) in enumerate(train_loader, start=start_batch):
            batch = batch.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs.logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            wandb.log({"d2_epoch": epoch, "d2_loss": loss.item()})

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

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, targets in val_loader:
                batch = batch.to(device)
                targets = targets.to(device)
                outputs = model(batch)
                loss = criterion(outputs.logits, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        wandb.log({"d2_val_loss": val_loss})

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epoch = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= patience:
                print('Early stopping...')
                return True

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1} completed, Avg Loss: {avg_loss}')

    return False

# Main script execution for pretraining
if os.path.isfile(model_path):
    print("Model already trained. Skipping training.")
else:
    # Extract frames from all mp4 files in subfolders within a directory
    video_files = glob.glob(local_data_dir + '/**/*.mp4', recursive=True)
    for video_file in tqdm(video_files, total=len(video_files), desc='Extracting frames'):
        extract_frames(video_file)

    num_files = int(len(glob.glob(local_data_dir + '/**/*.png', recursive=True)) * 0.01)
    print(f'Pretraining on {num_files} image files')

    # Prepare the pretraining data and initialize loaders
    train_loader, val_loader = prepare_pretraining_data(local_data_dir, num_files)

    print("Starting training process.")
    model, optimizer, scheduler, start_epoch, start_batch, best_loss = load_latest_checkpoint(model, optimizer, scheduler)
    train_loop(start_epoch, start_batch, best_loss)

# Fine-tuning with FathomNet dataset (continue with the same process)
class HierarchicalTaxonomyDataset(Dataset):
    def __init__(self, fathomnet_root_path, concepts, transform=None, non_taxonomic_labels=None):
        self.transform = transform
        self.image_dir = fathomnet_root_path
        self.concepts = concepts
        self.non_taxonomic_labels = non_taxonomic_labels or ["trash"]
        self.concept_to_index = {concept: i for i, concept in enumerate(concepts)}
        self.taxonomy_hierarchy = [self.get_taxonomy_hierarchy(concept) for concept in concepts]
        self.images_info = self.download_images_for_concepts(concepts)

    def get_taxonomy_hierarchy(self, species):
        if species.lower() in [label.lower() for label in self.non_taxonomic_labels]:
            return {"non_taxonomic": species}
        base_url = "https://api.gbif.org/v1/species/match"
        params = {"name": species, "verbose": True}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
            return {rank: data.get(rank, None) for rank in ranks}
        return None

    def download_images_for_concepts(self, concepts):
        image_data = []
        for concept in concepts:
            try:
                images_info = images.find_by_concept(concept)
                image_data.extend(images_info)
                for image_info in images_info:
                    image_url = image_info.url
                    image_path = os.path.join(self.image_dir, f"{image_info.uuid}.jpg")
                    if not os.path.exists(image_path):
                        self.download_image(image_url, image_path)
            except ValueError as ve:
                print(f"Error fetching image data for concept {concept}: {ve}")
                continue
        return image_data

    def download_image(self, url, save_path):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(save_path, 'wb') as handler:
                    handler.write(response.content)
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        try:
            image_info = self.images_info[idx]
            image_path = os.path.join(self.image_dir, f"{image_info.uuid}.jpg")
            image = Image.open(image_path).convert('RGB')
            labels_vector = torch.zeros(len(self.concepts))
            for box in image_info.boundingBoxes:
                if box.concept in self.concept_to_index:
                    labels_vector[self.concept_to_index[box.concept]] = 1
            taxonomy_labels = self.taxonomy_hierarchy[idx]
            if self.transform:
                image = self.transform(image)
            return image, labels_vector, taxonomy_labels
        except (IOError, OSError) as e:
            print(f"Error reading image {image_info.uuid}: {e}")
            return None, None, None

# Hierarchical prediction class for ViT model
class HierarchicalViT(nn.Module):
    def __init__(self, vit_model, threshold=0.5):
        super(HierarchicalViT, self).__init__()
        self.vit_model = vit_model
        self.threshold = threshold

    def forward(self, images):
        outputs = self.vit_model(images).logits
        softmax_outputs = torch.softmax(outputs, dim=-1)
        return self.predict_hierarchy(softmax_outputs)

    def predict_hierarchy(self, softmax_outputs):
        hierarchy_prediction = {}
        ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
        for i, rank in enumerate(ranks):
            rank_prediction = softmax_outputs[:, i].max(dim=-1)
            confidence = rank_prediction[0].item()
            predicted_class = rank_prediction[1].item()
            if confidence >= self.threshold:
                hierarchy_prediction[rank] = {
                    "predicted_class": predicted_class,
                    "confidence": confidence
                }
            else:
                break
        return hierarchy_prediction

def load_and_train_model(model_root_path, old_model_path, fathomnet_root_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    concepts = boundingboxes.find_concepts()
    dataset = HierarchicalTaxonomyDataset(fathomnet_root_path, concepts, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    vit_model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
    vit_model.classifier = nn.Linear(vit_model.config.hidden_size, len(concepts))
    hierarchical_vit = HierarchicalViT(vit_model, threshold=0.5).to(device)
    optimizer = optim.Adam(hierarchical_vit.parameters())
    num_epochs = 1
    patience = 2
    no_improve_epoch = 0
    save_freq = 1000
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=num_epochs)
    checkpoint_folder = os.path.join(model_root_path, 'fn_checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()

    def load_latest_checkpoint():
        checkpoints = glob.glob(os.path.join(checkpoint_folder, "*.pth"))
        checkpoints.sort(key=lambda x: [int(num) for num in re.findall(r'\d+', x)], reverse=True)
        if checkpoints:
            latest_checkpoint_path = checkpoints[0]
            checkpoint = torch.load(latest_checkpoint_path)
            hierarchical_vit.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch']
            best_loss = checkpoint['best_loss']
            print(f"Loaded Checkpoint from {latest_checkpoint_path}!!")
            return start_epoch, start_batch, best_loss
        else:
            return 0, 0, np.inf

    def train_loop(start_epoch, start_batch, best_loss):
        total_batches = len(train_loader)
        for epoch in range(start_epoch, num_epochs):
            print(f'Starting epoch {epoch + 1}/{num_epochs}')
            running_loss = 0.0
            hierarchical_vit.train()
            for batch_idx, (images, labels_vector) in enumerate(train_loader, start=start_batch):
                if images is None or labels_vector is None:
                    break
                images = images.to(device)
                labels_vector = labels_vector.to(device)
                optimizer.zero_grad()
                outputs = hierarchical_vit(images)
                loss = criterion(outputs, labels_vector)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                wandb.log({"fn_epoch": epoch, "fn_loss": loss.item()})
                if (batch_idx + 1) % save_freq == 0:
                    checkpoint_path = os.path.join(checkpoint_folder, f'fn_checkpoint_{epoch + 1}_{batch_idx + 1}.pth')
                    torch.save({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_state_dict': hierarchical_vit.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'best_loss': best_loss,
                    }, checkpoint_path)
            epoch_loss = running_loss / len(train_loader.dataset)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                print(f'New best loss: {best_loss}')
                no_improve_epoch = 0
            else:
                no_improve_epoch += 1
            if no_improve_epoch >= patience:
                break

    start_epoch, start_batch, best_loss = load_latest_checkpoint()
    train_loop(start_epoch, start_batch, best_loss)

final_model_path = os.path.join(model_root_path, 'fn_trained_model.pth')

if os.path.exists(final_model_path):
    print("Fully trained model already exists. Skipping training.")
else:
    old_model_path = os.path.join(model_root_path, 'd2_fine_tuned_model.pt')
    hierarchical_vit = load_and_train_model(model_root_path, old_model_path, fathomnet_root_path)
    torch.save(hierarchical_vit.state_dict(), final_model_path)
