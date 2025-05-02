# Configs and utility
import os
from datetime import datetime
from tqdm import tqdm 
import pandas as pd
import uuid
import requests
import json
from dotenv import load_dotenv

# geospatial libraries
import geopandas as gpd
from shapely.geometry import Polygon, Point

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

# Load environment variables
if os.path.exists('.env'):
    load_dotenv('.env', override=True, verbose=True)

# Get the environment variables
PORTAL_URL = os.getenv("PORTAL_URL")
WEB_SERVICE_URL = os.getenv("WEB_SERVICE_URL")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
YOLO_MODEL = os.getenv("YOLO_MODEL")
INPUT_CONNECTION_STRING = os.getenv("INPUT_CONNECTION_STRING")
INPUT_CONTAINER_NAME = os.getenv("INPUT_CONTAINER_NAME")
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD"))


"""Define utils functions"""
def authenticate(portal_url, username, password):
    token_url = f"{portal_url}/sharing/rest/generateToken"
    payload = {
        "username": username,
        "password": password,
        "referer": "https://www.arcgis.com",
        "f": "json",
    }
    response = requests.post(token_url, data=payload)
    if response.status_code == 200:
        token = response.json().get("token")
        if token:
            return token
        else:
            raise Exception(
                "Failed to generate token: "
                + response.json().get("error", {}).get("message", "Unknown error")
            )
    else:
        response.raise_for_status()


def insert_features_to_arcgis(url, features, token=None):
    add_features_url = f"{url}/addFeatures"

    payload = {
        "features": json.dumps(features),
        "f": "json",
    }

    if token:
        payload["token"] = token

    response = requests.post(add_features_url, data=payload)
    return response.json()

def query_features_from_arcgis(url, token, where_clause):
    query_url = f"{url}/query"
    params = {
        "where": where_clause,
        "outFields": "image_path",
        "f": "json"}
    if token:
        params["token"] = token
    response = requests.get(query_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def wrangling_geometry(coords):
    rings = []
    for coord in coords:
        ring = []
        for point in coord:
            point = list(point)
            ring.append(point)
        rings.append(ring)
    return {"rings": rings}


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Extracting the YOLO model name
    yolo_model_path = os.path.join("ml_models", YOLO_MODEL)
    yolo_model = YOLO(yolo_model_path)

    # Creating the connection
    blob_service_client = BlobServiceClient.from_connection_string(
        INPUT_CONNECTION_STRING
    )

    # Getting the container
    container_client = blob_service_client.get_container_client(INPUT_CONTAINER_NAME)

    # Listing all the blobs
    blobs = container_client.list_blobs()

    # Only leaving the images
    blobs = [blob for blob in blobs if is_blob_image(blob.name)]

    # Dropping the blobs that are in the 00_UNSORTED directory
    blobs = [blob for blob in blobs if "00_UNSORTED" not in blob.name]

    token = authenticate(PORTAL_URL, USERNAME, PASSWORD)
    # Querying the ArcGIS for the blobs that are already there
    arcgis_blobs = query_features_from_arcgis(
        url=WEB_SERVICE_URL, token=token, where_clause="1=1"
    )

    # arcgis_blobs = query_features_from_arcgis(
    #     url=WEB_SERVICE_URL, where_clause="1=1", token=None
    # )

    # Extracting the features from the ArcGIS response
    arcgis_blobs = arcgis_blobs.get("features", [])

    # Filtering out the blobs that are already in ArcGIS
    if arcgis_blobs:
        arcgis_blob_names = [feature["attributes"]["image_path"] for feature in arcgis_blobs]
        blobs = [blob for blob in blobs if blob.name not in arcgis_blob_names]

    # Creating the input directory to store the downloaded images
    input_dir = os.path.join(current_dir, "blob_input")
    os.makedirs(input_dir, exist_ok=True)

    blobs = blobs[0:10]

    # Placeholder for the parsed information
    sosnovskies, probabilities, classes = [] , [], []

    # Iterating over the blobs
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
            container=INPUT_CONTAINER_NAME, blob=blob.name
        )
        with open(path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        # Reading and ploting the image
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Predicting
        results = yolo_model.predict(img, conf=PREDICTION_THRESHOLD)
        segments = results[0].masks

        # Get probabilities and classes
        probs = results[0].boxes.conf
        mask_classes = results[0].boxes.cls

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
            for mask, prob, mask_cls in zip(segments, probs, mask_classes):
                points = mask.xy[0]
                for point in points:
                    sosnovskies.append(
                        (
                            f"{blob.name}",
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
                        f"{blob.name}",
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
                            relative_altitude,
                        ),
                    )
                )

                # Incrementing
                polygon_idx += 1

                probabilities.append(round(float(prob), 2))
                classes.append(int(mask_cls))

        # Clearning the image
        del img 
        os.remove(path)

    # Checking if there are any Sosnovskies found
    # if not finish pipeline
    if not sosnovskies:
        print("No Sosnovskies found in the images.")
        exit(0)

    # Create dataframe from the Sosnovskies list
    gps_points_df = pd.DataFrame(sosnovskies, columns=["image_path", "polygon_idx", "gps_coords"])
    gps_points_df['lat'] = gps_points_df['gps_coords'].apply(lambda x: x[0])
    gps_points_df['lon'] = gps_points_df['gps_coords'].apply(lambda x: x[1])

    # Dropping the gps_coords column
    gps_points_df.drop(columns=["gps_coords"], inplace=True)

    # Grouping the data by image path and polygon index
    gdf = gps_points_df.groupby(['image_path', 'polygon_idx']).agg(list).reset_index()

    # Assigning probabilities and classes to the dataframe
    gdf["prediction_prob"] = probabilities
    gdf['species'] = classes

    gdf["beginLifespanVersion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # gdf["begin_lifespan_version"] = gdf.apply(
    #     lambda x: pd.Timestamp(x["begin_lifespan_version"]).timestamp(), axis=1
    # )

    # Creating the geometry column
    gdf["geometry"] = gdf.apply(
        lambda row: Polygon([Point(xy) for xy in zip(row["lon"], row["lat"])]),
        axis=1,
    )

    # Creating the GeoDataFrame
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3346")
    gdf["population_size"] = round(gdf.geometry.area, 2)
    gdf['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gdf['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a unique ID for each prediction UUID
    gdf['prediction_id'] = gdf.apply(lambda x: str(uuid.uuid4()), axis=1)

    # Drop unnecessary columns
    gdf = gdf.drop(columns=["lat", "lon", "polygon_idx"])

    # Drop null polygon geometries
    gdf = gdf[gdf["population_size"] > 0.00]
    gdf.reset_index(drop=True, inplace=True)

    # Transform the GeoDataFrame to a dictionary format suitable for ArcGIS
    geojson_data = gdf.to_geo_dict(drop_id=True)

    # Get features from the GeoJSON data
    features = geojson_data.get("features", [])

    # Update features with the correct geometry format for ArcGIS
    agol_features = []
    for feature in features:
        # Get coordinates from the feature
        coords = feature["geometry"]["coordinates"]
        # Create geometry in ArcGIS format
        rings = wrangling_geometry(coords)
        # Create the ArcGIS feature
        agol_feature = {
            "geometry": rings,
            "attributes": feature["properties"],
        }
        # Append the feature to the list
        agol_features.append(agol_feature)

    # Insert features into ArcGIS
    token = authenticate(PORTAL_URL, USERNAME, PASSWORD)
    insert_features_to_arcgis(WEB_SERVICE_URL, agol_features, token)
    # insert_features_to_arcgis(WEB_SERVICE_URL,  agol_features)
