import os
import sys
import toml
import torch
import numpy as np
import random
import boto3
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from fathomnet.api import boundingboxes
import seaborn as sns

# Add the directory containing seabot_finetune.py to the Python path
sys.path.append("..")

# Import the necessary functions and classes from seabot_finetune.py
from seabot_finetune import FathomNetDataset, collate_fn, download_model_from_s3

# Load settings from the TOML configuration file
config = toml.load('analysis_settings.toml')
MODEL_NAME = config['settings']['MODEL_NAME']

# Configuration values
BUCKET_NAME = 'seabot-d2-storage'  # Replace with your actual S3 bucket name
S3_MODEL_ROOT_PATH = 'SeaBot/Test_Models/'  # Replace with your actual S3 model root path
FATHOMNET_RELATIVE_PATH = 'fathomnet'  # Replace with your actual FathomNet relative path
MODEL_ROOT_PATH = 'local_test_models'  # Replace with your actual local model root path

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Load and prepare the data
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
concepts = boundingboxes.find_concepts()  # Find the concepts for the bounding boxes
fathomnet_root_path = os.path.join(os.getcwd(), FATHOMNET_RELATIVE_PATH)
dataset = FathomNetDataset(fathomnet_root_path, concepts, transform=transform)

# Split the dataset into training and validation subsets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

# Download and load the model
s3_model_path = os.path.join(S3_MODEL_ROOT_PATH, f'{MODEL_NAME}_final.pth')
local_model_path = os.path.join(MODEL_ROOT_PATH, f'{MODEL_NAME}_final.pth')
download_model_from_s3(BUCKET_NAME, s3_model_path, local_model_path)
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224', num_labels=len(concepts))
model.load_state_dict(torch.load(local_model_path))
model.to(device)

# Evaluate the model and compute micro-averaged F1 score
def evaluate_model(model, val_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels_vector in val_loader:
            images = images.to(device)
            labels_vector = labels_vector.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs.logits).round()  # Convert logits to probabilities and round to nearest integer

            all_labels.append(labels_vector.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    precision, recall, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
    
    return f1_micro, all_labels, all_preds

f1_micro, all_labels, all_preds = evaluate_model(model, val_loader, device)
print(f"Micro-Averaged F1 Score: {f1_micro}")

# Save and upload the visualization
conf_matrix = confusion_matrix(all_labels.ravel(), all_preds.ravel())
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Save the image locally
analysis_local_path = os.path.join(MODEL_ROOT_PATH, 'Analysis')
os.makedirs(analysis_local_path, exist_ok=True)
local_image_path = os.path.join(analysis_local_path, 'confusion_matrix.png')
plt.savefig(local_image_path)

# Upload the image to S3
s3_client = boto3.client('s3')
analysis_s3_path = os.path.join(S3_MODEL_ROOT_PATH, 'Analysis', 'confusion_matrix.png')
s3_client.upload_file(local_image_path, BUCKET_NAME, analysis_s3_path)
print(f"Confusion matrix image uploaded to S3: {analysis_s3_path}")
