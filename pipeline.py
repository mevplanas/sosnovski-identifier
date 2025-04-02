# Configs and utility
import yaml 
import os
from tqdm import tqdm 

# Azure blobs
from azure.storage.blob import BlobServiceClient

# Deep learning library 
from ultralytics import YOLO 
import cv2

# Custom functions
from src.utils import (
    pixel_to_gps,
    calculate_gsd,
    is_blob_image,
    extract_gps_coordinates,
    get_relative_altidute,
    extract_focal_length
)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Reading the configuration file
    with open("configuration.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Extracting the YOLO model name  
    yolo_model_name = config["YOLO_MODEL"]
    yolo_model_path = os.path.join("ml_models", yolo_model_name)
    yolo_model = YOLO(yolo_model_path)

    # Extracting the input connection string and container name
    input_connection_string = config["INPUT_CONNECTION_STRING"]
    input_container_name = config["INPUT_CONTAINER_NAME"]

    # Creating the connection
    blob_service_client = BlobServiceClient.from_connection_string(
        input_connection_string
    )

    # Getting the container
    container_client = blob_service_client.get_container_client(input_container_name)

    # Listing all the blobs
    blobs = container_client.list_blobs()


     # Listing all the blobs
    blobs = container_client.list_blobs()

    # Only leaving the images
    blobs = [blob for blob in blobs if is_blob_image(blob.name)]

    # Dropping the blobs that are in the 00_UNSORTED directory
    blobs = [blob for blob in blobs if "00_UNSORTED" not in blob.name]

    # Creating the input directory to store the downloaded images 
    input_dir = os.path.join(current_dir, "blob_input")
    os.makedirs(input_dir, exist_ok=True)

    # Placeholder for the parsed information
    sosnovskies = []

    for blob in tqdm(blobs):
        # Getting the base name
        base_name = os.path.basename(blob.name)

        # Getting everything except the base name from blob name
        blob_dir = blob.name.replace(base_name, "")

        # Creating the directory
        os.makedirs(os.path.join(input_dir, blob_dir), exist_ok=True)

        # Defining the path to the image
        path = os.path.join(input_dir, blob_dir, base_name)

        # Downloading the blob
        blob_client = blob_service_client.get_blob_client(
            container=input_container_name, blob=blob.name
        )
        with open(path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        # Reading and ploting the image 
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


