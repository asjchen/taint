# Processes a raw image to resemble the emnist format
# Then, identifies the class of the resulting image
# Finally, runs the AdvGAN algorithm to modify the image for a targerted attack

import argparse
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from letter_classifier import LetterClassifier
from hyperparams import CLASSIFIER_CONFIGS

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

def process_color_image(image_filename, final_height=28, final_width=28):
    raw_grayscale = color_to_grayscale(image_filename)
    unnormed_grayscale = np.array(scale_and_crop_image(
        raw_grayscale, final_height, final_width))[:, :, 0].transpose()
    return normalize_numpy(unnormed_grayscale)

# Displays a NumPy array (with all elements <= 1) as a grayscale picture
def display_normed_image(normed_np):
    scaled_np = 256 * normed_np
    plt.imshow(scaled_np.transpose())
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description=('Takes an image and a '
        'classifier model and outputs an adversarial example'))
    parser.add_argument('image_filename', 
        help=('Filename of the (color) input image; the letter should be '
            'white on black background'))
    parser.add_argument('-d', '--display_image', action='store_true',
        help='Displays the grayscale scaled image')
    parser.add_argument('-a', '--architecture', 
        choices=CLASSIFIER_CONFIGS.keys(), default='cnn_two_layer',
        help=('Classifier architecture to be used, one of {}, default '
            'is cnn_two_layer'.format(list(CLASSIFIER_CONFIGS.keys()))))
    parser.add_argument('-c', '--checkpoint', default='tmp/model.ckpt',
        help=('File with classifier model checkpoint if the model has '
            'already been trained'))
    args = parser.parse_args()
    
    # Original input for the classifier, not for this program
    orig_input = process_color_image(args.image_filename)
    if args.display_image:
        display_normed_image(orig_input)

    config = CLASSIFIER_CONFIGS[args.architecture]
    with tf.Graph().as_default():
        classifier = LetterClassifier(config)
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, args.checkpoint)
            predicted_class = classifier.eval(session, np.array([orig_input]))
            print('Predicted Class: {}'.format(predicted_class))



if __name__ == '__main__':
    main()
