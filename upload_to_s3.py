import os
import sys
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: script.py model_name")
        sys.exit(1)

    model_name = sys.argv[1]
    model_file_name = f'{model_name}.pth'
    model_file_path = os.path.join(os.getcwd(), model_file_name)

    if os.path.exists(model_file_path):
        upload_to_s3(model_file_path, BUCKET_NAME, S3_MODEL_ROOT_PATH, model_name)
    else:
        print(f"Model file {model_file_name} not found in the current working directory.")
