# Processes a raw image to resemble the emnist format
# Then, identifies the class of the resulting image
# Finally, runs the AdvGAN algorithm to modify the image for a targerted attack

import argparse
import numpy as np
from PIL import Image

def color_to_grayscale(image_filename):
    gray_img = Image.open(image_filename).convert('LA')
    return gray_img

def normalize_numpy(img_256_arr):
    return img_256_arr / 255

def scale_and_crop_image(orig_img, final_height, final_width):
    if orig_img.size[0] * final_width != final_height * orig_img.size[1]:
        print('\nImage will be scaled to the size: {} x {}\n'.format(
            final_height, final_width))
    return orig_img.resize((final_height, final_width), Image.ANTIALIAS)

def process_color_image(image_filename, final_height, final_width):
    raw_grayscale = color_to_grayscale(image_filename)
    unnormed_grayscale = np.array(scale_and_crop_image(
        raw_grayscale, final_height, final_width))[:, :, 0].transpose()
    return normalize_numpy(unnormed_grayscale)

def main():
    parser = argparse.ArgumentParser(description=('Takes an image and a '
        'classifier model and outputs an adversarial example'))
    parser.add_argument('image_filename', 
        help='Filename of the (color) input image')
    args = parser.parse_args()
    # arguments to take in the image, whether to display the image
    

    # display it if asked
    # take in a model and identify the class (originally)

if __name__ == '__main__':
    main()
