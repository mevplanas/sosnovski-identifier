from PIL import Image as PILImage
import cv2 
import exifread
from exif import Image
import math

# Extracting the function to extract the focal length
def extract_focal_length(file_path):
    # Open the image file for reading in binary mode
    with open(file_path, 'rb') as f:
        # Read the EXIF data
        tags = Image(f.read())
        
        # Check if the image has EXIF data
        if not tags.has_exif:
            print("No EXIF data found in the image.")
            return None

        # Check if GPSInfo tag is present
        if 'focal_length' in dir(tags):
            # Extract latitude, longitude, and altitude
            focal_length = tags.focal_length

            return float(focal_length)
        else:
            print("Focal length not found in the metadata.")
            return None

def get_relative_altidute(img_path: str) -> float:
    """
    Descrption
    -----------
    The function extracts reltaive height from image metadata.

    Parameters
    ----------
    :param image_path : Path to the image

    Returns
    ----------
    :return: Relative height of the image
    """
    # Set default altitude to 0
    altitude = 0

    # Open the image using PIL
    with PILImage.open(img_path) as img:
        # Extract the XMP metadata
        try:
            # Get the XMP metadata from the image
            xmp = img.getxmp()
            # Get relative height from the XMP metadata
            altitude = xmp["xmpmeta"]["RDF"]["Description"]["RelativeAltitude"]
            # Convert the altitude to a float
            altitude = float(altitude)
        except Exception as e:
            print(f"Error extracting XMP metadata from image: {e}")

    # Extract the altitude value
    return altitude

def extract_gps_coordinates(file_path):
    # Open the image file for reading in binary mode
    with open(file_path, 'rb') as f:
        # Read the EXIF data
        tags = exifread.process_file(f)

        # Check if GPSInfo tag is present
        relative_altitude = 0
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags and 'GPS GPSAltitude' in tags:
            # Extract latitude, longitude, and altitude
            latitude = tags['GPS GPSLatitude'].values
            longitude = tags['GPS GPSLongitude'].values
            altitude = tags['GPS GPSAltitude'].values

            # Getting the relative altitude
            relative_altitude = get_relative_altidute(file_path)

            # Convert coordinates to decimal format
            latitude_decimal = latitude[0] + latitude[1] / 60 + latitude[2] / 3600
            longitude_decimal = longitude[0] + longitude[1] / 60 + longitude[2] / 3600

            # 
            return float(latitude_decimal), float(longitude_decimal), float(altitude[0]), float(relative_altitude)
        else:
            print("GPS information not found in the metadata.")
            return None, None, None, None

def pixel_to_gps(x, y, width, height, lat_center, lon_center, gsd_h, gsd_v, altitude):
    """
    Convert pixel coordinates to GPS coordinates for a nadir (straight-down) view,
    without considering gimbal orientation.
    
    Parameters:
      x, y          : Pixel coordinates
      width, height : Dimensions of the image (in pixels)
      lat_center    : Latitude of the center point (degrees)
      lon_center    : Longitude of the center point (degrees)
      gsd_h, gsd_v  : Horizontal and vertical ground sample distance (meters/pixel)
      altitude      : Camera altitude (meters) (not used in this simple model,
                      since gsd is assumed to be pre-calculated based on altitude)
    
    Returns:
      (lat, lon) : GPS coordinates of the pixel
    """
    # Offset from the image center in pixels
    dx = x - width / 2
    dy = y - height / 2

    # Convert pixel offsets to real-world distances in meters.
    # Note: Depending on your coordinate system, you might need to flip the sign of dy.
    delta_x_m = dx * gsd_h   # east-west displacement (meters)
    delta_y_m = dy * gsd_v   # north-south displacement (meters)

    # Convert meter offsets to degrees.
    # The conversion is approximate (suitable for small displacements).
    R_earth = 6371000  # Earth radius in meters
    delta_lat = (delta_y_m / R_earth) * (180 / math.pi)
    delta_lon = (delta_x_m / (R_earth * math.cos(math.radians(lat_center)))) * (180 / math.pi)

    # Calculate the final GPS coordinates
    lat = lat_center + delta_lat
    lon = lon_center + delta_lon

    return lat, lon

# Defining the function that calculates the diff in pixels to meters
# GSD - Ground Sampling Distance
def calculate_gsd(
        height_from_ground, 
        image_width, 
        image_height, 
        sensor_width, 
        sensor_height, 
        focal_length
        ):
    """
    Function that calculates the GSD (Ground Sampling Distance) from pixels to meters

    Args:
        height_from_ground (float): Height from ground in meters
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels
        sensor_width (float): Sensor width in mm
        sensor_height (float): Sensor height in mm
        focal_length (float): Focal length in mm

    Returns:
        gsd_h (float): Horizontal GSD in meters
        gsd_v (float): Vertical GSD in meters
    """
    # Calculating the horizontal and vertical GSD
    gsd_h = (height_from_ground * sensor_width) / (focal_length * image_width)
    gsd_v = (height_from_ground * sensor_height) / (focal_length * image_height)
    # Returning the average GSD
    return gsd_h, gsd_v

def is_blob_image(blob_name: str) -> bool:
    """
    Checks if the blob is an image
    """
    return (
        blob_name.endswith(".jpg")
        or blob_name.endswith(".png")
        or blob_name.endswith(".jpeg")
        or blob_name.endswith(".JPG")
        or blob_name.endswith(".PNG")
        or blob_name.endswith(".JPEG")
    )
