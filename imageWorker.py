from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS
import cv2
import numpy as np

def transpose_image(image_path):
    '''
    Takes input a path to an image and returns a transposed version based on EXIF data.
    '''
    with Image.open(image_path) as im:
        im_transposed = ImageOps.exif_transpose(im, False)
        return im_transposed

def get_exif_data(image_path):
    '''
    Takes input of a path to an image and returns all EXIF data in a dictionary
    '''
    # Open the image file
    with Image.open(image_path) as img:
        # get exif object
        exif_data = img._getexif()

        # case when no exif
        if exif_data is None:
            print("No EXIF data found.")
            return
        
        # populate the dictionary with tag names
        readable_exif = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            readable_exif[tag_name] = value
        
        return readable_exif
    
def auto_orient_image(image_path, threshold=10):
    '''
    Takes input a path to an image without EXIF metadata and returns an auto-oriented version the image using edge detection.
    Adjust the threshold value for the final affine transformation.
    '''
    # open the image
    img = cv2.imread(image_path)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # use Hough lines transformation
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # If lines were found, determine orientation
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            angles.append(angle)

        # Get average angle
        avg_angle = np.mean(angles)
        print(f"Average angle: {avg_angle}")

        # Rotate the image based on the average angle
        if abs(avg_angle) > threshold:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

    # Convert back to PIL
    oriented_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return oriented_image


