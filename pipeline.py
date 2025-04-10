# Configs and utility
import yaml 
import os
from tqdm import tqdm 
import pandas as pd

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
    extract_focal_length
)

# Hardcoding the ID Vilnius drone camera parameters 
SENSOR_WIDTH = 6.4
SENSOR_HEIGHT = 4.8

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

        # Predicting 
        results = yolo_model.predict(img, conf=0.05)
        segments = results[0].masks

        if segments is not None: 
            # Gathering all the needed information for the Sosnovskies
            img_w, img_h = img.shape[1], img.shape[0]
            focal_length = extract_focal_length(path)
            latitude, longitude, altitude, relative_altitude = extract_gps_coordinates(path)
            gsd_h, gsd_v = calculate_gsd(
                relative_altitude,
                img_w,
                img_h,
                SENSOR_WIDTH, 
                SENSOR_HEIGHT, 
                focal_length
                )

            if gsd_h is None or gsd_v is None:
                print(f"Skipping image {base_name} due to invalid GSD values.")
                continue
            
            # Creating the gps coordinates
            polygon_idx = 0
            for mask in segments:
                points = mask.xy[0]
                for point in points:
                    sosnovskies.append(
                        (
                            f"{blob_dir}/{base_name}",
                            polygon_idx,
                            pixel_to_gps(
                                point[0], 
                                point[1], 
                                img.shape[1], 
                                img.shape[0], 
                                latitude, 
                                longitude, 
                                gsd_h, 
                                gsd_v, 
                                relative_altitude
                            )
                        )
                    )
                # Adding an additional last point as the first point
                sosnovskies.append(
                    (
                        f"{blob_dir}/{base_name}",
                        polygon_idx,
                        pixel_to_gps(
                            points[0][0], 
                            points[0][1], 
                            img.shape[1], 
                            img.shape[0], 
                            latitude, 
                            longitude, 
                            gsd_h, 
                            gsd_v, 
                            relative_altitude
                        )
                    )
                )

                # Incrementing
                polygon_idx += 1

        # Clearning the image
        del img 
        os.remove(path)

    gps_points_df = pd.DataFrame(sosnovskies, columns=["image_path", "polygon_idx", "gps_coords"])
    gps_points_df['lat'] = gps_points_df['gps_coords'].apply(lambda x: x[0])
    gps_points_df['lon'] = gps_points_df['gps_coords'].apply(lambda x: x[1])
    gps_points_df.drop(columns=['gps_coords'], inplace=True)
