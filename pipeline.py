# Configs and utility
import os
import uuid
import argparse
from datetime import datetime, timedelta

import json
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# geospatial libraries
import geopandas as gpd
from geopy.distance import geodesic
from shapely.geometry import Polygon, Point

# Azure blobs
from azure.storage.blob import BlobServiceClient

# Deep learning library
from ultralytics import YOLO
import cv2

# Custom functions
from src.utils import pixel_to_gps, calculate_gsd, is_blob_image, extract_gps_coordinates, extract_focal_length

# Hardcoding the ID Vilnius drone camera parameters
SENSOR_WIDTH = 6.4
SENSOR_HEIGHT = 4.8

# Load environment variables
if os.path.exists(".env"):
    load_dotenv(".env", override=True, verbose=True)

# Get the environment variables
PORTAL_URL = os.getenv("PORTAL_URL")
PREDICTIONS_SERVICE_URL = os.getenv("PREDICTIONS_SERVICE_URL")
FORECAST_SERVICE_URL = os.getenv("FORECAST_SERVICE_URL")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
YOLO_MODEL = os.getenv("YOLO_MODEL")
INPUT_CONNECTION_STRING = os.getenv("INPUT_CONNECTION_STRING")
INPUT_CONTAINER_NAME = os.getenv("INPUT_CONTAINER_NAME")
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", 0.3))
METEO_API_KEY = os.getenv("METEO_API_KEY")


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
                "Failed to generate token: " + response.json().get("error", {}).get("message", "Unknown error")
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
    all_features = []
    for _ in range(0, 100000, 2000):
        params = {
            "where": where_clause,
            "outFields": "image_path",
            "resultOffset": _,
            "f": "json",
        }
        if token:
            params["token"] = token
        response = requests.get(query_url, params=params)
        if response.status_code == 200:
            arcgis_blobs = response.json()
            arcgis_blobs = arcgis_blobs.get("features", [])
            if not arcgis_blobs:
                return all_features
            all_features.extend(arcgis_blobs)

        else:
            response.raise_for_status()

    return all_features


def wrangling_geometry(coords):
    rings = []
    for coord in coords:
        ring = []
        for point in coord:
            point = list(point)
            ring.append(point)
        rings.append(ring)
    return {"rings": rings}


def gdf_to_arcgis(gdf: gpd.GeoDataFrame) -> None:
    """Transform the GeoDataFrame to a dictionary format suitable for ArcGIS"""
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

    return agol_features


def get_weather_data() -> dict:
    """
    Queries the weather data from the API
    """
    # Subtracting the current date by 1 day
    yesterday_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Formatting the date to be of the form YYYY-MM-DD
    formatted_date = datetime.strptime(yesterday_date, "%Y-%m-%d").strftime("%Y-%m-%d")

    # Querying the API
    response = requests.get(
        f"https://miesto-plauciai-functions.azurewebsites.net/api/meteoApi?code={METEO_API_KEY}&timestamp={formatted_date}"
    )

    output = {"wind_speed": 0.0, "wind_direction": 0.0}

    if response.status_code == 200:
        data = response.json()
        # Extracting the relevant data
        output["wind_speed"] = data["windSpeed"]  # M/S
        output["wind_direction"] = data["windDirection"]  # degrees; 0 from north, 90 from east

    return output


def push_point(lat, lon, bearing_deg, push_distance_m=5):
    # geodesic().destination expects km and bearing clockwise from north
    km = push_distance_m / 1000.0
    dest = geodesic(kilometers=km).destination((lat, lon), bearing_deg)

    return dest.latitude, dest.longitude


def df_to_gdf(df: pd.DataFrame, min_area: float) -> gpd.GeoDataFrame:
    """
    Description
    ---------
    Converts a DataFrame to a GeoDataFrame

    Parameters
    ----------
    :param df: DataFrame to convert
    :param min_area: Minimum area to keep the polygons

    Returns
    -------
    :return: GeoDataFrame with the polygons
    """
    # Creating the geometry column
    df["geometry"] = df.apply(lambda row: Polygon([Point(xy) for xy in zip(row["lon"], row["lat"])]), axis=1)

    # Creating the GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # Converting the GeoDataFrame to the desired CRS
    gdf = gdf.to_crs("EPSG:3346")

    # Calculating the area of the polygons
    gdf["populationSize"] = round(gdf.geometry.area, 2)

    # Dropping the polygons that are smaller than the minimum area
    gdf = gdf[gdf["populationSize"] > min_area]
    gdf.reset_index(drop=True, inplace=True)

    # Adding the created_at and updated_at columns
    gdf["created_at"] = int(datetime.now().timestamp()) * 1000
    gdf["updated_at"] = int(datetime.now().timestamp()) * 1000

    return gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # add true false argument for the pipeline
    parser.add_argument("--input_path", type=str, default="blob_input", help="Path to the input directory")
    parser.add_argument("--count", type=int, default=0, help="Number of images to process")
    parser.add_argument("--token", action="store_true", help="Run the pipeline")
    parser.add_argument("--no-token", action="store_false", help="Do not run the pipeline")

    parser.set_defaults(token=False)
    args = parser.parse_args()

    # Extracting the YOLO model name
    yolo_model_path = os.path.join("ml_models", YOLO_MODEL)
    yolo_model = YOLO(yolo_model_path)

    # Creating the connection
    blob_service_client = BlobServiceClient.from_connection_string(INPUT_CONNECTION_STRING)

    # Getting the current weather data
    weather_data = get_weather_data()
    wind_speed = weather_data["wind_speed"]
    wind_direction = weather_data["wind_direction"]

    # Getting the container
    container_client = blob_service_client.get_container_client(INPUT_CONTAINER_NAME)

    # Get container client url
    container_url = container_client.url

    # Listing all the blobs
    blobs = container_client.list_blobs()

    # Only leaving the images
    blobs = [blob for blob in blobs if is_blob_image(blob.name)]

    # Dropping the blobs that are in the 00_UNSORTED directory
    blobs = [blob for blob in blobs if "00_UNSORTED" not in blob.name]

    if args.token:
        token = authenticate(PORTAL_URL, USERNAME, PASSWORD)
    else:
        token = None

    # Querying the ArcGIS for the blobs that are already there
    arcgis_blobs = query_features_from_arcgis(url=PREDICTIONS_SERVICE_URL, token=token, where_clause="1=1")

    # Filtering out the blobs that are already in ArcGIS
    if arcgis_blobs:
        arcgis_blob_names = [feature["attributes"]["image_path"] for feature in arcgis_blobs]
        blobs = [blob for blob in blobs if blob.name not in arcgis_blob_names]

    # Creating the input directory to store the downloaded images
    input_dir = os.path.join(args.input_path)
    os.makedirs(input_dir, exist_ok=True)

    # Placeholder for the parsed information
    sosnovskies, probabilities, classes = [], [], []

    blobs = blobs[: args.count] if args.count > 0 else blobs

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
        blob_client = blob_service_client.get_blob_client(container=INPUT_CONTAINER_NAME, blob=blob.name)
        with open(path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        # Reading and ploting the image
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error reading image {base_name}: {e}")
            os.remove(path)
            continue

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
            try:
                latitude, longitude, altitude, relative_altitude = extract_gps_coordinates(path)
            except Exception as e:
                print(f"Error extracting GPS coordinates from {base_name}: {e}")
                os.remove(path)
                continue
            gsd_h, gsd_v = calculate_gsd(relative_altitude, img_w, img_h, SENSOR_WIDTH, SENSOR_HEIGHT, focal_length)

            if gsd_h is None or gsd_v is None:
                os.remove(path)
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
                                relative_altitude,
                            ),
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
    gps_points_df["lat"] = gps_points_df["gps_coords"].apply(lambda x: x[0])
    gps_points_df["lon"] = gps_points_df["gps_coords"].apply(lambda x: x[1])

    # Dropping the gps_coords column
    gps_points_df.drop(columns=["gps_coords"], inplace=True)

    # Creating another - forecasted polygon
    gps_points_df_forecasted = gps_points_df.copy()
    gps_points_df_forecasted["pushed"] = gps_points_df_forecasted.apply(
        lambda x: push_point(x["lat"], x["lon"], wind_direction), axis=1
    )
    gps_points_df_forecasted["lat"] = gps_points_df_forecasted["pushed"].apply(lambda x: x[0])
    gps_points_df_forecasted["lon"] = gps_points_df_forecasted["pushed"].apply(lambda x: x[1])
    gps_points_df_forecasted.drop(columns=["pushed"], inplace=True)

    # Grouping the data by image path and polygon index
    df = gps_points_df.groupby(["image_path", "polygon_idx"]).agg(list).reset_index()
    df_forecasted = gps_points_df_forecasted.groupby(["image_path", "polygon_idx"]).agg(list).reset_index()

    # TODO uncomment this when azure blob storage is ready
    # # Add image url
    # df["image_url"] = df.apply(lambda x: f"{container_url}/{x['image_path']}", axis=1)

    # Assigning probabilities and classes to the dataframe
    df["prediction_prob"] = probabilities
    df["species"] = classes

    # Add date of record creation as unix timestamp
    df["beginLifespanVersion"] = int(datetime.now().timestamp()) * 1000

    # Creating GeoDataFrames
    gdf = df_to_gdf(df, min_area=0.00)
    gdf_forecasted = df_to_gdf(df_forecasted, min_area=0.00)

    # Create a unique ID for each prediction UUID
    gdf["prediction_id"] = gdf.apply(lambda x: "{" + str(uuid.uuid4()) + "}", axis=1)

    # Assigning the prediction key to the forecasted GeoDataFrame
    gdf_forecasted["prediction_key"] = gdf["prediction_id"]

    # Drop unnecessary columns
    gdf = gdf.drop(columns=["lat", "lon", "polygon_idx"])
    gdf_forecasted = gdf_forecasted.drop(columns=["lat", "lon", "polygon_idx", "image_path", "populationSize"])

    # Transform the GeoDataFrame to a dictionary format suitable for ArcGIS
    agol_features = gdf_to_arcgis(gdf)
    agol_features_forecasted = gdf_to_arcgis(gdf_forecasted)

    # Generate token if needed
    if args.token:
        token = authenticate(PORTAL_URL, USERNAME, PASSWORD)
    else:
        token = None

    # Insert features into ArcGIS
    pred_response = insert_features_to_arcgis(PREDICTIONS_SERVICE_URL, agol_features, token)
    forecast_response = insert_features_to_arcgis(FORECAST_SERVICE_URL, agol_features_forecasted, token)

    # Print the responses
    print("Prediction response:", pred_response)
    print("Forecast response:", forecast_response)
