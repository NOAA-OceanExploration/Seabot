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

import boto3
import ffmpeg
import glob
import random
import numpy as np
import os
import torch
import wandb
import traceback
from PIL import Image
from random import randint
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from torchvision import transforms
from tqdm import tqdm

# Setup AWS S3
s3 = boto3.resource(
    service_name='s3',
    region_name='<your-region>',
    aws_access_key_id='<your-access-key-id>',
    aws_secret_access_key='<your-secret-access-key>'
)

bucket_name = '<your-bucket-name>'
prefix = 'SeaBot/Data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

wandb.init(project="seabot", config={
    "architecture": "vit-base-patch16-224",
    "dataset": "fathomnet & d2_dives",
})

def file_exists_in_s3(bucket_name, file_key):
    s3_client = boto3.client('s3')
    try:
        s3_client.head_object(Bucket=bucket_name, Key=file_key)
        return True
    except:
        return False

def check_image_integrity(image_key):
    try:
        temp_path = '/tmp/temp_image.png'
        s3.Bucket(bucket_name).download_file(image_key, temp_path)
        
        with Image.open(temp_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Corrupted file {image_key}: {e}")
        return False

checkpoint_folder = os.path.join(prefix, 'd2_checkpoints')
model_path = os.path.join(prefix, 'd2_fine_tuned_model.pt')

num_epochs = 2
patience = 2
save_freq = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
])

model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
model.classifier = nn.Linear(model.config.hidden_size, 4)
model = model.to(device)

optimizer = optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

criterion = nn.CrossEntropyLoss()

os.makedirs('/tmp/' + checkpoint_folder, exist_ok=True)  # Temporary local directory

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        temp_path = '/tmp/temp_image.png'
        s3.Bucket(bucket_name).download_file(self.image_files[idx], temp_path)
        image = Image.open(temp_path)
        rotation = randint(0, 3)
        rotated_image = image.rotate(rotation * 90)
        if self.transform:
            rotated_image = self.transform(rotated_image)
        return rotated_image, rotation

def extract_frames(video_path, frame_rate=1):
    out_path = video_path.rsplit('.', 1)[0]
    if file_exists_in_s3(bucket_name, out_path):
        return

    os.makedirs('/tmp/' + out_path, exist_ok=True)
    temp_video_path = '/tmp/temp_video.mp4'
    s3.Bucket(bucket_name).download_file(video_path, temp_video_path)

    (
        ffmpeg
        .input(temp_video_path)
        .filter('fps', fps=frame_rate)
        .output(os.path.join('/tmp/' + out_path, 'img%03d.png'))
        .run()
    )

    for file_name in glob.glob('/tmp/' + out_path + '/*.png'):
        s3.Bucket(bucket_name).upload_file(file_name, out_path + '/' + os.path.basename(file_name))

def load_latest_checkpoint(model, optimizer, scheduler):
    try:
        s3_client = boto3.client('s3')
        checkpoints = [obj.key for obj in s3.Bucket(bucket_name).objects.filter(Prefix=checkpoint_folder)]
        checkpoints.sort(key=lambda x: s3_client.head_object(Bucket=bucket_name, Key=x)['LastModified'])
        
        if checkpoints:
            latest_checkpoint_path = checkpoints[-1]
            print(f"Attempting to load checkpoint from {latest_checkpoint_path}")
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
            return model, optimizer, scheduler, start_epoch, start_batch, best_loss
        else:
            print("No Checkpoint found!!")
            return model, optimizer, scheduler, 0, 0, np.inf
    except Exception as e:
        print(f"Exception occurred: {e}")
        traceback.print_exc()
        raise e

# Define function to save a checkpoint
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    temp_path = '/tmp/' + filename
    torch.save(state, temp_path)
    s3.Bucket(bucket_name).upload_file(temp_path, prefix + '/' + filename)

def train_loop(start_epoch, start_batch, best_loss):
    print("Fine-tuning model")

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
                save_checkpoint({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

        scheduler.step()

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

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epoch = 0
            temp_model_path = '/tmp/temp_model.pth'
            torch.save(model.state_dict(), temp_model_path)
            s3.Bucket(bucket_name).upload_file(temp_model_path, model_path)
            print(f'New best model saved at epoch {epoch + 1}')
        else:
            no_improve_epoch += 1
            print(f'Patience counter: {no_improve_epoch}')
            if no_improve_epoch >= patience:
                print('Early stopping...')
                return True

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed, Average Loss: {avg_loss}')

    return False

video_files = [obj.key for obj in s3.Bucket(bucket_name).objects.filter(Prefix=prefix) if obj.key.endswith('.mp4')]

for video_file in tqdm(video_files, total=len(video_files), desc='Extracting frames'):
    extract_frames(video_file)

if file_exists_in_s3(bucket_name, model_path):
    print("The model has already been trained. Skipping training.")
else:
    image_files = [obj.key for obj in s3.Bucket(bucket_name).objects.filter(Prefix=prefix) if obj.key.endswith('.png')]

    print("Checking image integrity...")
    valid_files = [f for f in image_files if check_image_integrity(f)]

    if len(valid_files) < len(image_files):
        print(f"Found {len(image_files) - len(valid_files)} corrupted files. Proceeding with {len(valid_files)} valid files.")

    num_files = int(len(valid_files))
    selected_files = np.random.choice(valid_files, size=num_files, replace=False).tolist()

    train_files, val_files = train_test_split(selected_files, test_size=0.2, random_state=42)

    train_dataset = ImageDataset(train_files, transform)
    val_dataset = ImageDataset(val_files, transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    print("Starting the training process.")
    try:
        model, optimizer, scheduler, start_epoch, start_batch, best_loss = load_latest_checkpoint(model, optimizer, scheduler)
        train_loop(start_epoch, start_batch, best_loss)
    except Exception as e:
        print(f"Error occurred during training: {e}. Exiting...")
