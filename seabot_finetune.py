import argparse
import boto3
import glob
import os
import requests
import torch
import torchvision.transforms as transforms
import re
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from tqdm import tqdm
from transformers import ViTForImageClassification
import wandb
import numpy as np

# Helper Functions
def create_directory(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def s3_object_exists(s3_client, bucket_name, object_path):
    try:
        s3_client.head_object(Bucket=bucket_name, Key=object_path)
        return True
    except:
        return False

def transfer_s3_object(s3_client, bucket_name, source_path, dest_path, operation='download'):
    try:
        if operation == 'download':
            s3_client.download_file(bucket_name, source_path, dest_path)
        elif operation == 'upload':
            s3_client.upload_file(source_path, bucket_name, dest_path)
    except Exception as e:
        print(f"Error occurred during S3 {operation}: {e}")

# Define the custom dataset class for handling FathomNet data
class FathomNetDataset(Dataset):
    def __init__(self, fathomnet_root_path, concepts, s3_client, s3_bucket, transform=None, skip_download=False):
        self.transform = transform
        self.images_info = []  # Replace with actual code to fetch images info
        self.image_dir = fathomnet_root_path
        self.concepts = concepts
        self.concept_to_index = {concept: i for i, concept in enumerate(concepts)}
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.skip_download = skip_download

        for concept in concepts:
            # Replace with actual code to fetch images by concept
            pass

        self.images_info.sort(key=lambda x: x['uuid'])  # Sort based on a unique identifier

        create_directory(self.image_dir)

        self.download_and_process_images()

    def download_and_process_images(self):
        for image_info in tqdm(self.images_info, desc="Processing images", unit="image"):
            image_s3_path = f"{self.image_dir}/{image_info['uuid']}.jpg"
            image_path = os.path.join(self.image_dir, f"{image_info['uuid']}.jpg")

            if not self.skip_download and not s3_object_exists(self.s3_client, self.s3_bucket, image_s3_path):
                image_url = image_info['url']
                try:
                    image_data = requests.get(image_url).content
                    with open(image_path, 'wb') as handler:
                        handler.write(image_data)
                    transfer_s3_object(self.s3_client, self.s3_bucket, image_path, image_s3_path, 'upload')
                except Exception as e:
                    print(f"Error downloading image from {image_url}: {e}")
                    continue

            elif self.skip_download:
                transfer_s3_object(self.s3_client, self.s3_bucket, image_s3_path, image_path)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        image_path = os.path.join(self.image_dir, f"{image_info['uuid']}.jpg")
        image = Image.open(image_path).convert('RGB')

        labels_vector = torch.zeros(len(self.concepts))
        for box in image_info['boundingBoxes']:
            if box['concept'] in self.concept_to_index:
                labels_vector[self.concept_to_index[box['concept']]] = 1

        if self.transform:
            image = self.transform(image)

        return image, labels_vector

def collate_fn(batch):
    batch = [(image, label) for image, label in batch if image is not None and label is not None]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()
    images = torch.stack([item[0] for item in batch])
    labels_vector = torch.stack([item[1] for item in batch])
    return images, labels_vector

# Model Training Function
def train_model(args, s3_client, s3_bucket, model_root_path, fathomnet_root_path, concepts):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = FathomNetDataset(fathomnet_root_path, concepts, s3_client, s3_bucket, transform=transform, skip_download=args.skip_data_download)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

    model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
    model.classifier = nn.Linear(model.config.hidden_size, len(concepts))
    if args.continue_training and args.pretrained_model_path:
        model.load_state_dict(torch.load(args.pretrained_model_path))
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 1  # Define the number of epochs

    for epoch in range(num_epochs):
        model.train()
        for images, labels_vector in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels_vector)
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item()})

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, labels_vector in val_loader:
                outputs = model(images)
                loss = criterion(outputs.logits, labels_vector)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            wandb.log({"val_loss": val_loss})

    return model

# Main function to parse arguments and execute the script
def main():
    parser = argparse.ArgumentParser(description="Train image classification model on FathomNet data")
    parser.add_argument("--continue_training", action="store_true", help="Continue training from a pre-trained model")
    parser.add_argument("--pretrained_model_path", type=str, help="Path to the pre-trained model")
    parser.add_argument("--skip_data_download", action="store_true", help="Skip downloading data and use existing data")
    args = parser.parse_args()

    model_root_path = "SeaBot/FathomNet/Models"
    fathomnet_root_path = "SeaBot/FathomNet/Data"
    s3_client = boto3.client('s3')
    s3_bucket = 'seabot-d2-storage'
    concepts = []  # Replace with actual concepts

    create_directory(model_root_path)
    create_directory(fathomnet_root_path)

    wandb.login(key=WANDB_KEY)
    wandb.init(project="seabot", name="fn_finetuning")

    model = train_model(args, s3_client, s3_bucket, model_root_path, fathomnet_root_path, concepts)

    final_model_path = os.path.join(model_root_path, 'fn_trained_model.pth')
    s3_final_model_path = "SeaBot/FathomNet/Models/fn_trained_model.pth"
    if not s3_object_exists(s3_client, s3_bucket, s3_final_model_path):
        torch.save(model.state_dict(), final_model_path)
        transfer_s3_object(s3_client, s3_bucket, final_model_path, s3_final_model_path, 'upload')
        print(f"Model saved to S3 at {s3_final_model_path}")
    else:
        print("Model already exists in S3. Skipping save.")

if __name__ == "__main__":
    main()
