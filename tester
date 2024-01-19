import boto3
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification
from sklearn.metrics import average_precision_score
from fathomnet.api import images, boundingboxes
from PIL import Image
import requests
from tqdm import tqdm
import numpy as np

# Constants
BUCKET_NAME = 'seabot-d2-storage'
S3_MODEL_PATH = 'SeaBot/Models/fn_final_trained_model_pretrained.pth'
LOCAL_MODEL_PATH = 'fn_final_trained_model_pretrained.pth'
FATHOMNET_ROOT_PATH = 'path_to_fathomnet'  # Replace with your FathomNet root path

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to download the model from S3
def download_model_from_s3():
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(BUCKET_NAME, S3_MODEL_PATH, LOCAL_MODEL_PATH)
        print(f"Model downloaded from S3 to {LOCAL_MODEL_PATH}")
    except Exception as e:
        print(f"Error downloading model from S3: {e}")
        raise

# Define the custom dataset class for handling FathomNet test data
class FathomNetTestDataset(Dataset):
    def __init__(self, fathomnet_root_path, concepts, transform=None):
        self.transform = transform
        self.images_info = []
        self.image_dir = fathomnet_root_path
        self.concepts = concepts
        self.concept_to_index = {concept: i for i, concept in enumerate(concepts)}

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
        for image_info in tqdm(self.images_info, desc="Downloading test images", unit="image"):
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

            # Apply transformations if any
            if self.transform:
                image = self.transform(image)

            return image, labels_vector
        except (IOError, OSError):
            print(f"Error reading image {image_path}. Skipping.")
            return None, None

# Function to load the model
def load_model(num_labels):
    model = ViTForImageClassification.from_pretrained(
        'google/vit-large-patch16-224',
        num_labels=num_labels,
    )
    model.load_state_dict(torch.load(LOCAL_MODEL_PATH))
    model = model.to(device)
    model.eval()
    return model

# Function to evaluate the model
def evaluate_model(model, test_loader):
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for images, labels_vector in test_loader:
            if images is None or labels_vector is None:
                continue

            images = images.to(device)
            labels_vector = labels_vector.to(device)

            outputs = model(images).logits
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(labels_vector.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate mAP
    average_precision = average_precision_score(all_targets, all_outputs, average='macro')
    return average_precision

# Main
concepts = boundingboxes.find_concepts()  # Dynamically fetch the list of concepts
num_labels = len(concepts)

download_model_from_s3()
model = load_model(num_labels)

# Define the transformation for test data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prepare test dataset and loader
test_dataset = FathomNetTestDataset(FATHOMNET_ROOT_PATH, concepts, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=lambda batch: [(image, label) for image, label in batch if image is not None and label is not None])

# Evaluate the model
map_score = evaluate_model(model, test_loader)
print(f"Mean Average Precision (mAP): {map_score}")
