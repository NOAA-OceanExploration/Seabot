# Intended to generate fathomnet dataset within S3.

import os
import boto3
import requests
from tqdm import tqdm
from fathomnet.api import images, boundingboxes
from concurrent.futures import ThreadPoolExecutor

# AWS S3 setup
s3 = boto3.client('s3')
s3_bucket = 'your-s3-bucket-name'
s3_prefix = 'fathomnet-dataset/'  # S3 folder to store images

# Local path to temporarily store images before uploading
local_data_dir = '/tmp/fathomnet_dataset'
os.makedirs(local_data_dir, exist_ok=True)

# Function to download image from a URL
def download_image(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# Function to upload a file to S3
def upload_to_s3(local_file_path, s3_bucket, s3_key):
    try:
        s3.upload_file(local_file_path, s3_bucket, s3_key)
        print(f"Uploaded {local_file_path} to s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        print(f"Error uploading {local_file_path} to S3: {e}")

# Function to process a concept: fetch images, download, and upload them
def process_concept(concept):
    concept_dir = os.path.join(local_data_dir, concept)
    os.makedirs(concept_dir, exist_ok=True)

    # Fetch images from FathomNet for the given concept
    try:
        images_info = images.find_by_concept(concept)
    except Exception as e:
        print(f"Error fetching images for concept {concept}: {e}")
        return

    # Download and upload images
    for img_info in tqdm(images_info, desc=f"Processing {concept}"):
        image_url = img_info.url
        image_uuid = img_info.uuid
        image_file = f"{image_uuid}.jpg"
        local_image_path = os.path.join(concept_dir, image_file)
        s3_image_key = os.path.join(s3_prefix, concept, image_file)

        # Download image
        download_image(image_url, local_image_path)

        # Upload image to S3
        upload_to_s3(local_image_path, s3_bucket, s3_image_key)

        # Optionally, you can clean up the local image file to save space
        if os.path.exists(local_image_path):
            os.remove(local_image_path)

# Function to fetch and process multiple concepts in parallel
def process_fathomnet_concepts(concepts):
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_concept, concepts)

# Main entry point for the script
if __name__ == '__main__':
    # Fetch the list of concepts (species or taxa)
    try:
        concepts = boundingboxes.find_concepts()
    except Exception as e:
        print(f"Error fetching concepts: {e}")
        exit(1)

    # Process the fetched concepts and upload their images to S3
    process_fathomnet_concepts(concepts)
