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
    # Open the image using PIL
    with Image.open(img_path) as img:
        # Extract the XMP metadata
        xmp = img.getxmp()

    # Extract the relative altitude from the XMP metadata
    altitude = xmp["xmpmeta"]["RDF"]["Description"]["RelativeAltitude"]

    # Convert the altitude to a float
    altitude = float(altitude)

    return altitude
