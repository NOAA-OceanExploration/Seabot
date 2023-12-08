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
from sklearn.model_selection import train_test_split
import wandb

# Helper Functions
def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def check_frame_exists_s3(s3_client, bucket_name, frame_s3_path):
    try:
        s3_client.head_object(Bucket=bucket_name, Key=frame_s3_path)
        return True
    except:
        return False

def download_from_s3(s3_client, bucket_name, s3_file_path, local_file_path):
    s3_client.download_file(bucket_name, s3_file_path, local_file_path)

def upload_frame_to_s3(local_path, s3_path, s3_client, bucket_name):
    try:
        s3_client.upload_file(local_path, bucket_name, s3_path)
    except Exception as e:
        print(f"Error occurred while uploading {local_path} to S3: {e}")

# Define the custom dataset class for handling FathomNet data
class FathomNetDataset(Dataset):
    def __init__(self, fathomnet_root_path, concepts, s3_client, s3_bucket, transform=None, skip_download=False):
        self.transform = transform
        self.images_info = []
        self.image_dir = fathomnet_root_path
        self.concepts = concepts
        self.concept_to_index = {concept: i for i, concept in enumerate(concepts)}
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.skip_download = skip_download

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
        for image_info in tqdm(self.images_info, desc="Processing images", unit="image"):
            image_s3_path = f"{fathomnet_root_path}/{image_info.uuid}.jpg"
            image_path = os.path.join(self.image_dir, f"{image_info.uuid}.jpg")

            if self.skip_download or check_frame_exists_s3(self.s3_client, self.s3_bucket, image_s3_path):
                download_from_s3(self.s3_client, self.s3_bucket, image_s3_path, image_path)
            else:
                # Existing download code
                image_url = image_info.url
                try:
                    image_data = requests.get(image_url).content
                    with open(image_path, 'wb') as handler:
                        handler.write(image_data)
                    upload_frame_to_s3(image_path, image_s3_path, self.s3_client, self.s3_bucket)
                except ValueError as ve:
                    print(f"Error downloading image from {image_url}: {ve}")
                    continue

    def __len__(self):
        return len(self.images_info)

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

            if self.transform:
                image = self.transform(image)

            return image, labels_vector
        except (IOError, OSError):
            print(f"Error reading image {image_path}. Skipping.")
            return None, None

def collate_fn(batch):
    batch = [(image, label) for image, label in batch if image is not None and label is not None]
    if len(batch) == 0:
        return None, None
    images = torch.stack([item[0] for item in batch])
    labels_vector = torch.stack([item[1] for item in batch])
    return images, labels_vector

def load_and_train_model(model_root_path, pretrained_model_path, fathomnet_root_path, continue_training, s3_client, s3_bucket):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Fetch the concepts for the bounding boxes
    concepts = boundingboxes.find_concepts()

    dataset = FathomNetDataset(fathomnet_root_path, concepts, s3_client, s3_bucket, transform=transform, skip_download=not continue_training)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

    model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
    model.classifier = nn.Linear(model.config.hidden_size, len(concepts))
    if continue_training and pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path))
        print("Loaded the pretrained model parameters for further training")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    num_epochs = 1
    patience = 2
    no_improve_epoch = 0
    save_freq = 1000
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=num_epochs)
    checkpoint_folder = os.path.join(model_root_path, 'fn_checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)

    def load_latest_checkpoint():
        checkpoints = glob.glob(os.path.join(checkpoint_folder, "*.pth"))
        checkpoints.sort(key=lambda x: [int(num) for num in re.findall(r'\d+', x)], reverse=True)
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

    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    def train_loop(start_epoch, start_batch, best_loss):
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
                no_improve_epoch = 0
            else:
                no_improve_epoch += 1
            if no_improve_epoch >= patience:
                print(f'Early stopping after {patience} epochs without improvement.')
                break

    start_epoch, start_batch, best_loss = load_latest_checkpoint()
    train_loop(start_epoch, start_batch, best_loss)

final_model_path = os.path.join(model_root_path, 'fn_trained_model.pth')
s3_final_model_path = f"SeaBot/FathomNet/Models/fn_trained_model.pth"

# Save final model to S3 or local, depending on its existence in S3
if not check_frame_exists_s3(s3_client, s3_bucket, s3_final_model_path):
    torch.save(model.state_dict(), final_model_path)
    upload_frame_to_s3(final_model_path, s3_final_model_path, s3_client, s3_bucket)
    print(f"Model saved to S3 at {s3_final_model_path}")
else:
    print("Model already exists in S3. Skipping save.")

# Argument Parsing
parser = argparse.ArgumentParser(description="Train image classification model on FathomNet data")
parser.add_argument("--continue_training", action="store_true", help="Continue training from a pre-trained model")
parser.add_argument("--pretrained_model_path", type=str, help="Path to the pre-trained model")
parser.add_argument("--skip_data_download", action="store_true", help="Skip downloading data and use existing data")
args = parser.parse_args()

# S3 Configuration
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
s3_bucket = 'seabot-d2-storage'

# Paths Configuration
model_root_path = "SeaBot/FathomNet/Models"
fathomnet_root_path = "SeaBot/FathomNet/Data"

# Ensure necessary directories exist
create_directory(model_root_path)
create_directory(fathomnet_root_path)

# Wandb Configuration
wandb.init(project="fathomnet-training", entity="your_wandb_entity")

# Load and Train Model
if not args.skip_data_download or not check_frame_exists_s3(s3_client, s3_bucket, f"{fathomnet_root_path}/"):
    print("Starting data processing and model training...")
    load_and_train_model(model_root_path, args.pretrained_model_path, fathomnet_root_path, args.continue_training, s3_client, s3_bucket)
else:
    print("Skipping data download. Using existing data for training.")
