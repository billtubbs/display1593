import pickle
from pathlib import Path

import numpy as np
from PIL import Image

# mask1593[i] holds the flat (256x256) pixel indices averaged to produce
# the colour of LED i, padded with 0 to a fixed width of 45 columns. 0 is
# never a genuine sample (the nearest any LED gets to pixel (0, 0) is
# (1, 1); the closest real index anywhere in the mask is 296), so it's
# safe to use as a padding sentinel and exclude from the average below.
# Pickled under Python 2 (protocol 0), hence encoding="latin1".
_DATA_DIR = Path(__file__).parent / "data"
with open(_DATA_DIR / "mask1593.pickle", "rb") as f:
    mask1593 = pickle.load(f, encoding="latin1")
_mask1593_valid = mask1593 != 0


def prepare_image(image, size=(256, 256)):
    """Crop image to a centred square and resize it for convert_image()."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")
    width, height = image.size
    if width != height:
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    return np.asarray(image.resize(size))


def convert_image(image_array):
    """Convert a 256x256 RGB image array to 1593 RGB LED intensities."""
    pixels = image_array.reshape(-1, image_array.shape[-1])
    samples = pixels[mask1593].astype(float)
    samples[~_mask1593_valid] = np.nan
    return np.nanmean(samples, axis=1).astype(int)
