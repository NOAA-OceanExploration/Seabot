import os
import boto3
from botocore.exceptions import NoCredentialsError

BUCKET_NAME = settings.BUCKET_NAME
S3_MODEL_ROOT_PATH = settings.S3_MODEL_ROOT_PATH

def upload_to_s3(local_file, bucket, s3_path, model_name):
    """
    Upload a model file to an S3 bucket

    :param local_file: Path to the local model file
    :param bucket: Name of the S3 bucket
    :param s3_path: S3 path where the model should be saved
    :param model_name: Name of the model file
    :return: True if file was uploaded, else False
    """
    s3 = boto3.client('s3')
    s3_model_path = os.path.join(s3_path, f'{model_name}.pth')

    try:
        s3.upload_file(local_file, bucket, s3_model_path)
        print(f"File {local_file} uploaded to {bucket}/{s3_model_path}")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def find_and_upload_model(model_name):
    """
    Find a model file in the current working directory and upload it to S3

    :param model_name: Name of the model file (without extension)
    """
    current_working_dir = os.getcwd()
    model_file_name = f'{model_name}.pth'
    model_file_path = os.path.join(current_working_dir, model_file_name)

    if os.path.exists(model_file_path):
        upload_to_s3(model_file_path, BUCKET_NAME, S3_MODEL_ROOT_PATH, model_name)
    else:
        print(f"Model file {model_file_name} not found in the current working directory.")

# Example usage
model_name = 'your_model_file_name'  # Replace with your model file name without extension
find_and_upload_model(model_name)
