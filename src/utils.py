from PIL import Image


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
    with Image.open(img_path) as img:
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
