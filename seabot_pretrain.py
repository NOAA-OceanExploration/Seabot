import argparse
import boto3
import ffmpeg
import glob
import numpy as np
import os
import random
import torch
import traceback
from PIL import Image
from tqdm import tqdm
from random import randint
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import hashlib
import wandb

# Import Dynaconf and load configurations
from dynaconf import Dynaconf

# Initialize settings
settings = Dynaconf(settings_files=['setting.toml'])

# Replace hardcoded values with configuration values
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WANDB_KEY = settings.WANDB_KEY
BUCKET_NAME = settings.BUCKET_NAME
NUM_EPOCHS = settings.NUM_EPOCHS
PATIENCE = settings.PATIENCE
SAVE_FREQ = settings.SAVE_FREQ
LOCAL_MODEL_DIR = settings.LOCAL_MODEL_DIR
LOCAL_VIDEO_DIR = settings.LOCAL_VIDEO_DIR
LOCAL_IMAGE_DIR = settings.LOCAL_IMAGE_DIR
DATASET_ROOT_PATH = settings.DATASET_ROOT_PATH
IMAGE_ROOT_PATH = settings.IMAGE_ROOT_PATH
MODEL_ROOT_PATH = settings.MODEL_ROOT_PATH
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
MODEL_NAME = settings.MODEL_NAME
safe_model_name = MODEL_NAME.replace('/', '_')

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Generate a dynamic run name based on the model name
run_name = f"drive_pretraining_{safe_model_name}_EX2304"

wandb.login(key=WANDB_KEY)
wandb.init(project="seabot", name=run_name)

# Helper Functions
def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def check_s3_path(s3, bucket, path):
    bucket_obj = s3.Bucket(bucket)
    objects = list(bucket_obj.objects.filter(Prefix=path))
    if len(objects) > 0 and objects[0].key == path:
        return True
    else:
        return False

def download_from_s3(s3_client, bucket_name, s3_file_path, local_file_path):
    s3_client.download_file(bucket_name, s3_file_path, local_file_path)

def check_frame_exists_s3(s3_client, bucket_name, frame_s3_path):
    try:
        s3_client.head_object(Bucket=bucket_name, Key=frame_s3_path)
        return True
    except:
        return False

def generate_frame_hash(video_file_name, frame_number):
    # Create a unique hash for each frame using the video file name and frame number
    frame_id = f"{video_file_name}-{frame_number}"
    return hashlib.md5(frame_id.encode()).hexdigest()

def upload_frame_to_s3(local_path, s3_path, s3_client, bucket_name):
    try:
        s3_client.upload_file(local_path, bucket_name, s3_path)
    except Exception as e:
        print(f"Error occurred while uploading {local_path} to S3: {e}")

def extract_frames(bucket_name, video_s3_path, local_video_path, frame_rate=1):
    local_video_file = os.path.join(local_video_path, os.path.basename(video_s3_path))
    video_file_name = os.path.splitext(os.path.basename(local_video_file))[0]
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, video_s3_path, local_video_file)
    
    # Extract frames to LOCAL_IMAGE_DIR
    ffmpeg.input(local_video_file).filter('fps', fps=frame_rate).output(f"{LOCAL_IMAGE_DIR}/{video_file_name}_frame%03d.png").run()

    # Process each extracted frame
    frame_paths = glob.glob(f"{LOCAL_IMAGE_DIR}/{video_file_name}_frame*.png")
    for frame_path in frame_paths:
        frame_number = int(os.path.basename(frame_path)[len(video_file_name) + 6:-4])  # Extract frame number from filename
        frame_hash = generate_frame_hash(video_file_name, frame_number)
        s3_frame_path = f"{IMAGE_ROOT_PATH}/{frame_hash}.png"

        # Check if frame already exists in S3
        if not check_frame_exists_s3(s3_client, bucket_name, s3_frame_path):
            # Upload frame to S3 with error handling
            upload_frame_to_s3(frame_path, s3_frame_path, s3_client, bucket_name)
            # Optionally delete the frame from local storage after upload
        os.remove(frame_path)
            
def load_latest_checkpoint(model, optimizer, scheduler):
    try:
        checkpoints = glob.glob(os.path.join(LOCAL_MODEL_DIR, "*.pth"))
        checkpoints.sort(key=os.path.getmtime)
        if checkpoints:
            latest_checkpoint_path = checkpoints[-1]
            checkpoint = torch.load(latest_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['batch'], checkpoint['best_loss']
        else:
            return model, optimizer, scheduler, 0, 0, np.inf
    except Exception as e:
        print(f"Exception occurred: {e}")
        traceback.print_exc()
        raise e

def save_model_to_s3(local_model_path, s3_model_path, bucket_name):
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_model_path, bucket_name, s3_model_path)
        print(f"Model successfully uploaded to {s3_model_path} in bucket {bucket_name}")
    except Exception as e:
        print(f"Error occurred while uploading model to S3: {e}")

def train_loop(start_epoch, start_batch, best_loss, model, optimizer, scheduler, train_loader, val_loader, criterion):
    global PATIENCE
    no_improve_epoch = 0
    best_val_loss = best_loss

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            # Get model outputs
            outputs = model(images)

            # If the output is a tensor, use it directly; otherwise, use the logits attribute
            if isinstance(outputs, torch.Tensor):
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs.logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Wandb logging for training batch loss
            wandb.log({"epoch": epoch, "batch": batch_idx, "train_batch_loss": loss.item()})

            if (batch_idx + 1) % SAVE_FREQ == 0:
                checkpoint_path = os.path.join(LOCAL_MODEL_DIR, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
                torch.save({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_val_loss,
                }, checkpoint_path)

        avg_train_loss = total_loss / len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(DEVICE), val_labels.to(DEVICE)
                val_outputs = model(val_images)
                batch_loss = criterion(val_outputs.logits, val_labels)
                val_loss += batch_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Wandb logging for average training and validation loss
        wandb.log({"epoch": epoch, "avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss})

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_filename = f'best_model_{safe_model_name}.pth'  # Include model name in the filename
            best_model_path = os.path.join(LOCAL_MODEL_DIR, best_model_filename)
            torch.save(model.state_dict(), best_model_path)

            # Save to S3
            s3_model_path = os.path.join(MODEL_ROOT_PATH, best_model_filename)
            save_model_to_s3(best_model_path, s3_model_path, BUCKET_NAME)

            no_improve_epoch = 0
        else:
            no_improve_epoch += 1

        # Early stopping check
        if no_improve_epoch >= PATIENCE:
            print("Early stopping due to no improvement in validation loss.")
            break

        # Learning rate scheduler step
        scheduler.step()

        print(f'Epoch {epoch+1}: Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')


def list_s3_files(bucket_name, prefix, extension):
    s3_client = boto3.client('s3')
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    # Iterate through each page of results
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for item in page['Contents']:
                if item['Key'].endswith(extension):
                    files.append(item['Key'])

    return files

# Create necessary directories
create_directory(LOCAL_MODEL_DIR)
create_directory(LOCAL_VIDEO_DIR)
create_directory(LOCAL_IMAGE_DIR)

# Initialize S3 session
s3 = boto3.resource('s3')

# Define model, optimizer, scheduler, and criterion
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=4)  # Adjust num_classes as per your dataset
for param in model.parameters():
    param.requires_grad = True
model = model.to(DEVICE)
optimizer = optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Define the Dataset class
class ImageDataset(Dataset):
    def __init__(self, s3_client, bucket_name, image_keys, transform=None):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.image_keys = image_keys
        self.transform = transform

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        image_key = self.image_keys[idx]
        image_path = os.path.join(LOCAL_IMAGE_DIR, os.path.basename(image_key))
        self.s3_client.download_file(self.bucket_name, image_key, image_path)

        with Image.open(image_path) as img:
            rotation = randint(0, 3)
            rotated_image = img.rotate(rotation * 90)
            if self.transform:
                rotated_image = self.transform(rotated_image)

        # Delete the image from local storage after processing
        os.remove(image_path)

        return rotated_image, rotation

# Main script logic
skip_extract = True
if not os.path.isfile(LOCAL_MODEL_DIR):
    if not skip_extract:
        video_files = list_s3_files(BUCKET_NAME, DATASET_ROOT_PATH, extension='.mp4')
        for video_file in tqdm(video_files, desc='Extracting frames'):
            extract_frames(BUCKET_NAME, video_file, LOCAL_VIDEO_DIR)

    s3_client = boto3.client('s3')
    image_keys = list_s3_files(BUCKET_NAME, IMAGE_ROOT_PATH, extension='.png')  # Ensure this function returns the list of keys

    # Randomly select 10% of the image keys
    # selected_keys = random.sample(image_keys, k=int(0.1 * len(image_keys)))
    
    # Split the selected keys into training and validation sets
    train_keys, val_keys = train_test_split(image_keys, test_size=0.2, random_state=0)

    # Initialize the datasets
    train_dataset = ImageDataset(s3_client, BUCKET_NAME, train_keys, TRANSFORM)
    val_dataset = ImageDataset(s3_client, BUCKET_NAME, val_keys, TRANSFORM)

    # Initialize the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    print("Starting the training process.")
    try:
        model, optimizer, scheduler, start_epoch, start_batch, best_loss = load_latest_checkpoint(model, optimizer, scheduler)
        train_loop(start_epoch, start_batch, best_loss, model, optimizer, scheduler, train_loader, val_loader, criterion)
    except Exception as e:
        print(f"Error occurred during training: {e}. Exiting...")
