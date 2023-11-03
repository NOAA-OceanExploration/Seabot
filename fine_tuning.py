"""
Description:
This script defines and utilizes a custom dataset class `FathomNetDataset` to handle data from FathomNet for fine-tuning a pre-trained
Vision Transformer model. The main steps include data preparation, dataset splitting, model loading, fine-tuning, and checkpoint handling. 
A custom collate function `collate_fn` is used to handle batches with missing data. The `load_and_train_model` function orchestrates the
entire process including dataset creation, data loading, model preparation, training loop, and checkpoint saving/loading. Finally, the
script sets paths and invokes the `load_and_train_model` function to execute the fine-tuning process, and saves the fine-tuned model 
to disk.

Usage:
1. Set the `fathomnet_root_path` and `model_root_path` to the respective directories.
2. Set `old_model_path` to the path of the model to be fine-tuned.
3. Run the script to load the data, prepare the model, and initiate the fine-tuning process.
4. The fine-tuned model parameters are saved to 'fn_trained_model.pth' in the `model_root_path` directory.

Parameters:
- fathomnet_root_path (str): Root path to the FathomNet data.
- model_root_path (str): Root path to the directory for saving/loading models.
- old_model_path (str): Path to the pre-trained model for fine-tuning.
"""

import boto3
import glob
import numpy as np
import os
import re
import requests
import torch
import wandb
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification
from tqdm import tqdm

# Setup AWS S3
s3 = boto3.resource(
    service_name='s3',
    region_name='your-region',
    aws_access_key_id='your-access-key-id',
    aws_secret_access_key='your-secret-access-key'
)
bucket_name = 'your-bucket-name'
prefix = 'FathomNet/Data'

# Define the custom dataset class for handling FathomNet data
class FathomNetDataset(Dataset):
    def __init__(self, concepts, transform=None):
        self.transform = transform
        self.images_info = []
        self.concepts = concepts
        self.concept_to_index = {concept: i for i, concept in enumerate(concepts)}

        print("Number of classes in set: " + str(len(concepts)))

        # Fetch image data for each concept and save the information
        for concept in concepts:
            # Adjust this part to fetch data from your data source
            pass

        # Sort images info to ensure consistent order across different runs
        self.images_info.sort(key=lambda x: x.uuid)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        temp_path = '/tmp/temp_image.jpg'
        image_info = self.images_info[idx]
        image_key = f"{prefix}/{image_info.uuid}.jpg"
        s3.Bucket(bucket_name).download_file(image_key, temp_path)
        image = Image.open(temp_path).convert('RGB')

        # Create label vector
        labels_vector = torch.zeros(len(self.concepts))
        for box in image_info.boundingBoxes:
            if box.concept in self.concept_to_index:
                labels_vector[self.concept_to_index[box.concept]] = 1

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, labels_vector

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

def load_and_train_model(model_root_path, old_model_path):
    # Define a transformation that resizes images to 224x224 pixels and then converts them to tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Find the concepts for the bounding boxes
    concepts = boundingboxes.find_concepts()

    # Create a dataset with the given concepts and the defined transform
    dataset = FathomNetDataset(concepts, transform=transform)

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
    temp_old_model_path = '/tmp/temp_old_model.pth'
    s3.Bucket(bucket_name).download_file(old_model_path, temp_old_model_path)
    model.load_state_dict(torch.load(temp_old_model_path))
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
    checkpoint_folder = os.path.join(prefix, 'fn_checkpoints')

    # Make sure the checkpoint folder exists
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Load the latest checkpoint if it exists
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_folder, "*.pth")))

    # Define a function to load the latest checkpoint
    def load_latest_checkpoint():
        s3_client = boto3.client('s3')
        checkpoints = [obj.key for obj in s3.Bucket(bucket_name).objects.filter(Prefix=checkpoint_folder)]
        checkpoints.sort(key=lambda x: s3_client.head_object(Bucket=bucket_name, Key=x)['LastModified'])

        if checkpoints:
            latest_checkpoint_path = checkpoints[-1]
            temp_checkpoint_path = '/tmp/temp_checkpoint.pth'
            s3.Bucket(bucket_name).download_file(latest_checkpoint_path, temp_checkpoint_path)
            checkpoint = torch.load(temp_checkpoint_path)
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
                    temp_checkpoint_path = '/tmp/temp_checkpoint.pth'
                    torch.save({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'best_loss': best_loss,
                    }, temp_checkpoint_path)
                    s3.Bucket(bucket_name).upload_file(temp_checkpoint_path, checkpoint_path)
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

# Set model_root_path and old_model_path as parameters to be passed to the function
old_model_path = os.path.join(prefix, 'd2_fine_tuned_model.pt')
model_root_path = prefix
load_and_train_model(model_root_path, old_model_path)

# Save the fine-tuned model to S3
temp_model_path = '/tmp/fn_trained_model.pth'
torch.save(model.state_dict(), temp_model_path)
s3.Bucket(bucket_name).upload_file(temp_model_path, os.path.join(model_root_path, 'fn_trained_model.pth'))
