"""## Fine-tuning Image Vision Model (videomae-base)"""
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
import os
import re
import requests
import torch
import traceback

# Setup for using GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import wandb
wandb.login(key='your_wandb_key_here')

wandb.init(
    project="seabot",
    name=f"drive_pretraining_2",
)

dataset_root_path = "/content/drive/MyDrive/Colab Notebooks/Work/SeaBot/Data"
fathomnet_root_path = "drive/MyDrive/Colab Notebooks/Work/SeaBot/FathomNet"
model_root_path = "drive/MyDrive/Colab Notebooks/Work/SeaBot"

if os.path.exists(dataset_root_path):
    print('The directory exists')
else:
    print('The directory does not exist')

"""### Dive Pretraining"""

# Define paths
checkpoint_folder = os.path.join(model_root_path, 'd2_checkpoints')
model_path = os.path.join(model_root_path, 'd2_fine_tuned_model.pt')

# Define hyperparameters
num_epochs = 5
patience = 2
save_freq = 100

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

# Unfreeze all layers for training
for param in model.parameters():
    param.requires_grad = True

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
        checkpoints.sort(key=os.path.getmtime)  # Sorting based on last modification time

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
        print(f'Epoch {epoch + 1} completed, Average Loss: {avg_loss}')

    return False  # Indicate that training is not finished

if os.path.isfile(model_path):
    print("The model has already been trained. Skipping training.")
else:
    # Extract frames from all mp4 files in subfolders within a directory
    video_files = glob.glob(dataset_root_path + '/**/*.mp4', recursive=True)

    for video_file in tqdm(video_files, total=len(video_files), desc='Extracting frames'):
        extract_frames(video_file)

    image_files = glob.glob(dataset_root_path + '/**/*.png', recursive=True)
    # Train on a percentage of dive images
    num_files = int(len(image_files) * 0.01)
    print('pretraining on ' + str(num_files) + ' files')

    selected_files = np.random.choice(image_files, size=num_files, replace=False).tolist()

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

class HierarchicalTaxonomyDataset(Dataset):
    def __init__(self, fathomnet_root_path, concepts, transform=None, non_taxonomic_labels=None):
        """
        A dataset class to handle images and their hierarchical taxonomic labels (e.g., kingdom, phylum, etc.).
        Args:
            fathomnet_root_path (str): Path to the root directory where FathomNet images are stored.
            concepts (list): List of species or taxonomic concepts to be used for fetching images.
            transform (torchvision.transforms): Transformations to apply to the images.
            non_taxonomic_labels (list): List of labels that do not belong to any taxonomy (e.g., 'trash').
        """
        self.transform = transform
        self.image_dir = fathomnet_root_path
        self.concepts = concepts
        self.non_taxonomic_labels = non_taxonomic_labels or ["trash"]  # Default to "trash" if no other non-taxonomic labels
        self.concept_to_index = {concept: i for i, concept in enumerate(concepts)}  # Concept-to-index mapping

        # Fetch taxonomic hierarchy for each concept
        self.taxonomy_hierarchy = [self.get_taxonomy_hierarchy(concept) for concept in concepts]

        # Fetch image data for each concept and store it
        self.images_info = self.download_images_for_concepts(concepts)

    def get_taxonomy_hierarchy(self, species):
        """
        Fetches the taxonomic hierarchy for a given species name using the GBIF API or marks non-taxonomic labels.
        Args:
            species (str): Species name or concept.
        Returns:
            dict: A dictionary containing the hierarchical taxonomic labels (kingdom, phylum, etc.), or None if non-taxonomic.
        """
        # Check if the label is in the non-taxonomic list (e.g., 'trash')
        if species.lower() in [label.lower() for label in self.non_taxonomic_labels]:
            return {"non_taxonomic": species}  # Return a special marker for non-taxonomic labels

        base_url = "https://api.gbif.org/v1/species/match"
        params = {"name": species, "verbose": True}
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if data.get("matchType") == "NONE":
                return None  # Species not found
            # Extract taxonomic information
            ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
            taxonomy = {rank: data.get(rank, None) for rank in ranks}
            return taxonomy
        return None

    def download_images_for_concepts(self, concepts):
        """
        Fetches image information for each concept and downloads the images if they don't already exist.
        Args:
            concepts (list): List of species or taxonomic concepts.
        Returns:
            list: A list of image information for each concept.
        """
        image_data = []
        for concept in concepts:
            try:
                # Assuming FathomNet API is used to get images by concept
                images_info = images.find_by_concept(concept)
                image_data.extend(images_info)

                # Download images
                for image_info in images_info:
                    image_url = image_info.url
                    image_path = os.path.join(self.image_dir, f"{image_info.uuid}.jpg")

                    if not os.path.exists(image_path):  # Download if the image doesn't exist
                        self.download_image(image_url, image_path)

            except ValueError as ve:
                print(f"Error fetching image data for concept {concept}: {ve}")
                continue

        return image_data

    def download_image(self, url, save_path):
        """
        Downloads an image from a given URL and saves it to the specified path.
        Args:
            url (str): URL of the image to download.
            save_path (str): Path where the image will be saved.
        """
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(save_path, 'wb') as handler:
                    handler.write(response.content)
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.images_info)

    def __getitem__(self, idx):
        """
        Fetches an image and its hierarchical taxonomy labels at a given index.
        Args:
            idx (int): Index of the image to fetch.
        Returns:
            tuple: (image, label_vector, taxonomy_labels), where taxonomy_labels is a dictionary of hierarchical labels.
        """
        try:
            # Fetch image path and open the image
            image_info = self.images_info[idx]
            image_path = os.path.join(self.image_dir, f"{image_info.uuid}.jpg")
            image = Image.open(image_path).convert('RGB')

            # Fetch label vector for multi-label classification (assuming bounding box info is provided)
            labels_vector = torch.zeros(len(self.concepts))
            for box in image_info.boundingBoxes:
                if box.concept in self.concept_to_index:
                    labels_vector[self.concept_to_index[box.concept]] = 1

            # Fetch hierarchical taxonomy labels for the image
            taxonomy_labels = self.taxonomy_hierarchy[idx]

            # Apply transformations to the image
            if self.transform:
                image = self.transform(image)

            return image, labels_vector, taxonomy_labels

        except (IOError, OSError) as e:
            print(f"Error reading image {image_info.uuid}: {e}")
            return None, None, None

# Define hierarchical prediction logic with confidence threshold
class HierarchicalViT(nn.Module):
    def __init__(self, vit_model, threshold=0.5):
        """
        Wraps a ViT model to implement hierarchical taxonomy prediction with confidence threshold.
        Args:
            vit_model (ViTForImageClassification): Pre-trained Vision Transformer model.
            threshold (float): Confidence threshold for hierarchical prediction.
        """
        super(HierarchicalViT, self).__init__()
        self.vit_model = vit_model
        self.threshold = threshold

    def forward(self, images):
        """
        Forward pass for hierarchical prediction.
        Args:
            images (Tensor): Input images.
        Returns:
            dict: Predictions at each hierarchical level with their confidence scores.
        """
        outputs = self.vit_model(images).logits
        softmax_outputs = torch.softmax(outputs, dim=-1)  # Get softmax probabilities for each class

        return self.predict_hierarchy(softmax_outputs)

    def predict_hierarchy(self, softmax_outputs):
        """
        Predicts taxonomic hierarchy based on confidence scores and threshold.
        Args:
            softmax_outputs (Tensor): Softmax probabilities for each class.
        Returns:
            dict: Dictionary of predictions up to the level where confidence exceeds the threshold.
        """
        hierarchy_prediction = {}
        ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

        for i, rank in enumerate(ranks):
            rank_prediction = softmax_outputs[:, i].max(dim=-1)
            confidence = rank_prediction[0].item()  # Get the max probability (confidence)
            predicted_class = rank_prediction[1].item()  # Get the predicted class

            if confidence >= self.threshold:
                hierarchy_prediction[rank] = {
                    "predicted_class": predicted_class,
                    "confidence": confidence
                }
            else:
                break  # Stop predicting further down the hierarchy if confidence is below the threshold

        return hierarchy_prediction

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
    dataset = HierarchicalTaxonomyDataset(fathomnet_root_path, concepts, transform=transform)

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
    vit_model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
    vit_model.classifier = nn.Linear(vit_model.config.hidden_size, 4)

    # Unfreeze all layers for training
    for param in vit_model.parameters():
        param.requires_grad = True

    # Load the pre-trained model parameters for further training
    vit_model.load_state_dict(torch.load(old_model_path))
    print("Loaded the d2 model parameters for further training")

    # Replace the classifier again, this time with the number of concept classes
    vit_model.classifier = nn.Linear(vit_model.config.hidden_size, len(concepts))

    # Wrap the model in HierarchicalViT for hierarchical prediction
    hierarchical_vit = HierarchicalViT(vit_model, threshold=0.5).to(device)

    # Define the optimizer as Adam
    optimizer = optim.Adam(hierarchical_vit.parameters())

    # Define the number of training epochs and the patience for early stopping
    num_epochs = 1
    patience = 2
    no_improve_epoch = 0

    # Frequency for saving the model
    save_freq = 1000

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
            hierarchical_vit.load_state_dict(checkpoint['model_state_dict'])
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
    hierarchical_vit.to(device)

    # Define a function for the training loop
    def train_loop(start_epoch, start_batch, best_loss):
        total_batches = len(train_loader)  # Total number of batches in one epoch
        for epoch in range(start_epoch, num_epochs):
            print(f'Starting epoch {epoch + 1}/{num_epochs}')
            running_loss = 0.0
            hierarchical_vit.train()

            for batch_idx, (images, labels_vector) in enumerate(train_loader, start=start_batch):
                if images is None or labels_vector is None:
                    print("Terminating batch due to image or label vector read error.")
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

                # Print epoch progress
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Epoch {epoch + 1} Progress: {progress:.2f}%")

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

final_model_path = os.path.join(model_root_path, 'fn_trained_model.pth')

if os.path.exists(final_model_path):
    print("Fully trained model already exists. Skipping training.")
else:
    old_model_path = os.path.join(model_root_path, 'd2_fine_tuned_model.pt')
    hierarchical_vit = load_and_train_model(model_root_path, old_model_path, fathomnet_root_path)
    torch.save(hierarchical_vit.state_dict(), final_model_path)
