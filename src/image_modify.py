# Processes a raw image to resemble the emnist format
# Then, identifies the class of the resulting image
# Finally, runs the AdvGAN algorithm to modify the image for a targerted attack

import argparse
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def color_to_grayscale(image_filename):
    gray_img = Image.open(image_filename).convert('LA')
    return gray_img

def normalize_numpy(img_256_arr):
    return img_256_arr / 255.0

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

# Displays a NumPy array (with all elements <= 1) as a grayscale picture
def display_normed_image(normed_np):
    scaled_np = 256 * normed_np
    plt.imshow(scaled_np)
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description=('Takes an image and a '
        'classifier model and outputs an adversarial example'))
    parser.add_argument('image_filename', 
        help='Filename of the (color) input image')
    parser.add_argument('-d', '--display_image', action='store_true',
        help='Displays the grayscale scaled image')
    args = parser.parse_args()
    
    
    # TODO: change 28 -- user should choose among several settings,
    #  each w/ different JSON hyperparams
    # Original input for the classifier, not for this program
    orig_input = process_color_image(args.image_filename, 28, 28)
    if args.display_image:
        display_normed_image(orig_input)
        
    # take in a model and identify the class (originally)

if __name__ == '__main__':
    main()
