import numpy as np
import imageio
from PIL import Image


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h # the width and height after cropped
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return np.array(Image.fromarray(x[j:j+crop_h, i:i+crop_w].astype(np.uint8)).resize([resize_w, resize_w]))


def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.  # change pixel value range from [0,255] to [-1,1] to feed into CNN


def inverse_transform(images):
    return (images+1.)/2. # change image pixel value(outputs from tanh in range [-1,1]) back to [0,1]


def imread(path, is_grayscale = False):
    if (is_grayscale):
        return imageio.imread(path, flatten = True).astype(np.float) # [width,height] flatten RGB image to grayscale image
    else:
        return imageio.imread(path).astype(np.float) # [width,height,color_dim]


def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

