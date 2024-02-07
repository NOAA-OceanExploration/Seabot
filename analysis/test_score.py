import os
import sys
import toml
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from fathomnet.api import boundingboxes

# Add the directory containing seabot_finetune.py to the Python path
sys.path.append("..")

# Import the necessary functions and classes from seabot_finetune.py
from seabot_finetune import FathomNetDataset, collate_fn, download_model_from_s3, evaluate_model

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
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

# Download and load the model
s3_model_path = os.path.join(S3_MODEL_ROOT_PATH, f'{MODEL_NAME}_final.pth')
local_model_path = os.path.join(MODEL_ROOT_PATH, f'{MODEL_NAME}_final.pth')
download_model_from_s3(BUCKET_NAME, s3_model_path, local_model_path)
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224', num_labels=len(concepts))
model.load_state_dict(torch.load(local_model_path))
model.to(device)

# Evaluate the model
accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
