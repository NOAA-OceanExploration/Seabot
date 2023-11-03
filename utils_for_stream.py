"""
This script is designed for video processing and classification tasks. It's structured into several sections:

1. ImageDataset Class: 
    A custom dataset class to handle image data loading and transformations.

2. Frame Extraction and Classification:
    Functions for extracting frames from a video and classifying them using a pre-trained model.

3. GPT-3 Humanization:
    Functions to humanize the classification results using the OpenAI GPT-3 model.

4. Real-Time Variant:
    Functions and setups for real-time video processing and classification.

5. Real-Time Variant + Chat Log:
    Extensions to the real-time variant, with chat log posting capabilities.

6. SeaBot - Video:
    Setup for a different model (ViViT) to handle video data, along with training loops and classification logic.

7. SeaBot - Real Time:
    Real-time classification using a webcam or video stream, leveraging the trained ViViT model.

Dependencies:
    - PyTorch, torchvision
    - PIL
    - OpenAI GPT-3
    - cv2 (OpenCV)
    - pafy
    - requests
    - glob
    - (and others as imported in the script)

Usage:
    Adjust the file paths, model paths, and other configuration settings as per the requirements, and run the script for video processing and classification tasks. Ensure to have the necessary dependencies installed, and API keys set up for GPT-3 interactions.
    
"""


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def extract_frames(video_path, frame_rate=1):
    # use ffmpeg to extract frames from the video
    pass  # Fill this in

def classify_frames(video_path, model, frame_rate=1):
    # Extract frames from the video
    extract_frames(video_path, frame_rate)

    # Load frames
    frame_dir = video_path.rsplit('.', 1)[0]
    frame_files = sorted(glob.glob(frame_dir + '/*.png'))
    frame_dataset = ImageDataset(frame_files, transform)
    frame_loader = DataLoader(frame_dataset, batch_size=1)  # Classify one frame at a time

    # Classify frames
    model.eval()
    seen_classes = set()
    first_spotted = {}
    with torch.no_grad():
        for i, frame in enumerate(frame_loader):
            frame = frame.to(device)
            output = model(frame.unsqueeze(0))
            _, predicted = torch.max(output.logits.data, 1)
            predicted_class = predicted.item()

            if predicted_class not in seen_classes:
                seen_classes.add(predicted_class)
                timecode = i / frame_rate  # Calculate timecode
                first_spotted[concepts[predicted_class]] = timecode  # concepts is a list of all the concepts

    return first_spotted


# Make sure you have the right API key
openai.api_key = ''

def humanize_classification(entity, timecode):
    # Make a prompt for the GPT-3 model
    prompt = f"The entity '{entity}' was first spotted at timecode '{timecode}'. How would a human casually say this?"

    # Call the GPT-3 API
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=60)

    # Return the humanized result
    return response.choices[0].text.strip()

# Specify the path to save the results
result_path = "/content/drive/MyDrive/Colab Notebooks/Work/SeaBot/Results/result.txt"

# Specify the paths to the video files
dataset_root_path = "/content/drive/MyDrive/Colab Notebooks/Work/SeaBot/Data/EX2205"
video_files = glob.glob(dataset_root_path + '/*.mp4')
video_files = video_files[0:1]

# Load the trained model
model_path = os.path.join(model_root_path, 'fn_fine_tuned_model.pt')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.classifier = nn.Linear(model.config.hidden_size, len(concepts))
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# Set transforms for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize images
    transforms.ToTensor(),  # convert to tensor
])

# Classify the frames in each video and write the results to a text file
with open(result_path, "w") as file:
    for video_file in video_files:
        first_spotted = classify_frames(video_file, model)
        for entity, timecode in first_spotted.items():
            humanized_result = humanize_classification(entity, timecode)
            file.write(f"{video_file}, {humanized_result}\n")

"""### Real Time Variant"""

import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
import openai

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def process_video_stream(video_path, model, device, transform):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    model.eval()
    seen_classes = set()
    first_spotted = {}

    with torch.no_grad():
        i = 0
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Convert the frame to PIL Image and apply transformations
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if transform:
                    image = transform(image)

                # Classify the frame
                image = image.to(device)
                output = model(image.unsqueeze(0))
                _, predicted = torch.max(output.logits.data, 1)
                predicted_class = predicted.item()

                if predicted_class not in seen_classes:
                    seen_classes.add(predicted_class)
                    timecode = i / frame_rate  # Calculate timecode
                    first_spotted[concepts[predicted_class]] = timecode

                i += 1
            else:
                break

    # When everything done, release the video capture object
    cap.release()

    return first_spotted

openai.api_key = ''

def humanize_classification(entity, timecode):
    # Make a prompt for the GPT-3 model
    prompt = f"The entity '{entity}' was first spotted at timecode '{timecode}'. How would a human casually say this?"

    # Call the GPT-3 API
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=60)

    # Return the humanized result
    return response.choices[0].text.strip()

result_path = "/content/drive/MyDrive/Colab Notebooks/Work/SeaBot/Results/result.txt"

dataset_root_path = "/content/drive/MyDrive/Colab Notebooks/Work/SeaBot/Data/EX2205"
video_files = glob.glob(dataset_root_path + '/*.mp4')
video_files = video_files[0:1]

model_path = os.path.join(model_root_path, 'fn_fine_tuned_model.pt')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.classifier = nn.Linear(model.config.hidden_size, len(concepts))
model.load_state_dict(torch.load(model_path))
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize images
    transforms.ToTensor(),  # convert to tensor
])

with open(result_path, "w") as file:
    for video_file in video_files:
        first_spotted = process_video_stream(video_file, model, device, transform)
        for entity, timecode in first_spotted.items():
            humanized_result = humanize_classification(entity, timecode)
            file.write(f"{video_file}, {humanized_result}\n")

"""## Real Time Variant + Chat Log"""

import os
import cv2
import pafy
import torch
from PIL import Image
from torchvision import transforms
import openai
from ScsBot import ScsBot

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def process_video_stream(video_path, model, device, transform):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    model.eval()
    seen_classes = set()
    first_spotted = {}

    with torch.no_grad():
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if transform:
                    image = transform(image)
                image = image.to(device)
                output = model(image.unsqueeze(0))
                _, predicted = torch.max(output.logits.data, 1)
                predicted_class = predicted.item()

                if predicted_class not in seen_classes:
                    seen_classes.add(predicted_class)
                    timecode = i / frame_rate
                    first_spotted[concepts[predicted_class]] = timecode
                i += 1
            else:
                break
    cap.release()
    return first_spotted

openai.api_key = ''

def humanize_classification(entity, timecode):
    prompt = f"The entity '{entity}' was first spotted at timecode '{timecode}'. How would a human casually say this?"
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=60)
    humanized_result = response.choices[0].text.strip()
    chatbot.post_chat_msg(humanized_result)
    return humanized_result

jid = "your_jid"
password = "your_password"
room = "your_room"
nick = "your_nickname"
post_interval = 10
valid_data_regex = 'your_regex'

chatbot = ScsBot(jid, password, room, nick, post_interval, valid_data_regex)

if chatbot.connect((chathost, chatport)):
    chatbot.process(threaded=True)
else:
    print("Couldn't connect to chatroom")

result_path = "/content/drive/MyDrive/Colab Notebooks/Work/SeaBot/Results/result.txt"
dataset_root_path = "/content/drive/MyDrive/Colab Notebooks/Work/SeaBot/Data/EX2205"

# Get the video URL
url = "https://www.youtube.com/watch?v=your_livestream_id"
video = pafy.new(url)
best = video.getbest(preftype="mp4")  # you may need to adjust this depending on the stream

# Replace video_files with the URL of the YouTube livestream
video_files = [best.url]

model_path = os.path.join(model_root_path, 'fn_fine_tuned_model.pt')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.classifier = nn.Linear(model.config.hidden_size, len(concepts))
model.load_state_dict(torch.load(model_path))
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

with open(result_path, "w") as file:
    for video_file in video_files:
        first_spotted = process_video_stream(video_file, model, device, transform)
        for entity, timecode in first_spotted.items():
            humanized_result = humanize_classification(entity, timecode)
            file.write(f"{video_file}, {humanized_result}\n")

"""# SeaBot - Video"""

#! pip install einops
#! git clone https://github.com/drv-agwl/ViViT-pytorch.git
#! cp ViViT-pytorch/models.py .
#! cp ViViT-pytorch/main.py .

import torch
import torch.nn.functional as F
from models import ViViTBackbone
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import cv2
import numpy as np
import os

import copy
from torch.optim.lr_scheduler import StepLR

# Define the custom dataset class for handling FathomNet data
class FathomNetDataset(Dataset):
    def __init__(self, concepts, transform=None):
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

        # Create directory if it doesn't exist
        os.makedirs(self.image_dir, exist_ok=True)

        # Download images for each image info and save it to disk
        for image_info in tqdm(self.images_info, desc="Downloading images", unit="image"):
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

    # Get the number of images in the dataset
    def __len__(self):
        return len(self.images_info)

    # Fetch an image and its label vector by index
    def __getitem__(self, idx):
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

# Add normalization to transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Fetch concepts and create dataset with updated transform
concepts = boundingboxes.find_concepts()
dataset = FathomNetDataset(concepts, transform=transform)

# Splitting the dataset into training and validation datasets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
# Set a seed for the random number generator
torch.manual_seed(0)

# Split the dataset into training and validation subsets
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Instantiate the model
model = ViViTBackbone(
    t=1,  # Number of frames
    h=224,  # Height of frames
    w=224,  # Width of frames
    patch_t=1,  # Temporal patch size
    patch_h=16,  # Height patch size
    patch_w=16,  # Width patch size
    num_classes=len(concepts),
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=2048,
    model=0  # Model variant
)

# Change the number of output units of the model to match the number of classes in the dataset
# model.fc = nn.Linear(2048, len(concepts))

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the learning rate scheduler
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

model = model.to(device)

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

# Define the training loop
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            dataloader = train_loader
        else:
            model.eval()   # Set model to evaluate mode
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'fsa_transformer_best_model.pth')

import os
import openai
import requests
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from torch import nn
from ViViT import ViViTBackbone

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def extract_frames(video_path, frame_rate=1):
    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    count = 0
    while success:
        if count % frame_rate == 0:
            cv2.imwrite(f"frame{count}.jpg", image)
        success, image = video.read()
        count += 1
    video.release()

def classify_frames(video_path, model, frame_rate=1):
    # Extract frames from the video
    extract_frames(video_path, frame_rate)

    # Load frames
    frame_files = sorted(glob.glob('*.jpg'))
    frame_dataset = ImageDataset(frame_files, transform)
    frame_loader = DataLoader(frame_dataset, batch_size=1)  # Classify one frame at a time

    # Classify frames
    model.eval()
    seen_classes = set()
    first_spotted = {}
    with torch.no_grad():
        for i, frame in enumerate(frame_loader):
            frame = frame.to(device).unsqueeze(0)  # Add an extra dimension for num_frames
            output = model(frame)
            _, predicted = torch.max(output.data, 1)
            predicted_class = predicted.item()

            if predicted_class not in seen_classes:
                seen_classes.add(predicted_class)
                timecode = i / frame_rate  # Calculate timecode
                first_spotted[concepts[predicted_class]] = timecode  # concepts is a list of all the concepts

    return first_spotted

def humanize_classification(entity, timecode):
    # Make a prompt for the GPT-3 model
    prompt = f"The entity '{entity}' was first spotted at timecode '{timecode}'. How would a human casually say this?"

    # Call the GPT-3 API
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=60)

    # Return the humanized result
    return response.choices[0].text.strip()

# Make sure you have the right API key
openai.api_key = 'your_openai_api_key'

# Specify the path to save the results
result_path = "result.txt"

# Specify the paths to the video files
dataset_root_path = "/path/to/your/videos"
video_files = glob.glob(dataset_root_path + '/*.mp4')

# Load the trained model
model_path = os.path.join(dataset_root_path, 'best_model.pth')
v = ViViTBackbone(
    t=1,  # Number of frames
    h=224,  # Height of frames
    w=224,  # Width of frames
    patch_t=1,  # Temporal patch size
    patch_h=16,  # Height patch size
    patch_w=16,  # Width patch size
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=2048,
    model=0  # Model variant
)
num_ftrs = v.fc.in_features
v.fc = nn.Linear(num_ftrs, len(concepts))
v.load_state_dict(torch.load(model_path))
v = v.to(device)

# Set transforms for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize images
    transforms.ToTensor(),  # convert to tensor
])

# Classify the frames in each video and write the results to a text file
with open(result_path, "w") as file:
    for video_file in video_files:
        first_spotted = classify_frames(video_file, v)
        for entity, timecode in first_spotted.items():
            humanized_result = humanize_classification(entity, timecode)
            file.write(f"{video_file}, {humanized_result}\n")

"""# Seabot - Real Time"""

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with your video stream URL

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize images
    transforms.ToTensor(),  # convert to tensor
])

v.eval()  # Make sure model is in evaluation mode

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    image = Image.fromarray(frame).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = v(image.unsqueeze(0))  # Unsqueeze to add artificial first dimension
        _, predicted = torch.max(output.data, 1)
        predicted_class = predicted.item()

    print('Predicted:', concepts[predicted_class])

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()