# Processes a raw image to resemble the EMNIST format (28x28 grayscale)
# Then, identifies the class of the resulting image
# Finally, runs an adversary to modify the image for a targerted attack

import arg_utils
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from letter_classifier import LetterClassifier
from hyperparams import CLASSIFIER_CONFIGS, ADVERSARY_CONFIGS
from grad_adv import GradAdv

def color_to_grayscale(image_filename):
    """
    Opens a color image and converts it to grayscale.

    Args:
        image_filename: the filename of the color image

    Returns:
        A PIL.Image.Image object, representing the grayscale image

    """
    gray_img = Image.open(image_filename).convert('LA')
    return gray_img

def scale_image(orig_img, final_height, final_width):
    """
    Scales an image to a specified height and width. The Image.ANTIALIAS
    option means the original image won't be cropped before scaling, so the 
    new image may look "stretched" compared to the original image. 

    Args:
        orig_img: a PIL.Image.Image object, representing the original 
        (grayscale) image

        final_height: the height of the scaled image

        final_width: the width of the scaled image

    Returns:
        A PIL.Image.Image object, representing the scaled image.

    """
    if orig_img.size[0] * final_width != final_height * orig_img.size[1]:
        print('\nImage will be scaled to the size: {} x {}\n'.format(
            final_height, final_width))
    return orig_img.resize((final_height, final_width), Image.ANTIALIAS)

def normalize_numpy(img_256_arr):
    """
    Normalizes an image NumPy array so its values lie in the range [0, 1]

    Args:
        img_256_arr: a NumPy array (intended to be 2D or 3D) whose values lie 
        in the range [0, 255], representing an image

    Returns:
        A NumPy array, with the same dimensions as the input, whose values lie 
        in [0, 1]

    """
    return img_256_arr / 255.0

def process_color_image(image_filename, final_height, final_width):
    """
    Reads, desaturates, scales, converts, and normalizes a color image, so 
    that it can be used as input for the adversary.

    Args:
        image_filename: the filename of the color (unscaled) image

        final_height: the height of the scaled image

        final_width: the width of the scaled image

    Returns:
        A 2D NumPy array, whose values are in [0, 1], representing the scaled 
        grayscale image

    """
    raw_grayscale = color_to_grayscale(image_filename)
    unnormed_grayscale = np.array(scale_image(raw_grayscale, final_height, 
        final_width))[:, :, 0].transpose()
    return normalize_numpy(unnormed_grayscale)

def display_normed_image(normed_np):
    """
    Displays a NumPy array (with all elements in [0, 1]) as a grayscale picture

    Args:
        normed_np: a 2D NumPy array, whose elements are in [0, 1], 
        representing a grayscale image

    Returns:
        None

    """
    scaled_np = 255.0 * normed_np
    plt.imshow(scaled_np.transpose())
    plt.show()
    plt.close()

def make_output_directory(output_directory):
    """
    Creates the directory to place the tainted image if it doesn't yet exist

    Args:
        output_directory: a string denoting the output directory path

    Returns:
        None
        
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

def make_output_image_filename(output_directory, 
    predicted_letter, target_letter):
    """
    Creates the path for the output tainted image: if the classifier 
    categorizes the original image as the letter "p" and the tainted image as 
    the letter "t", then the filename is <output_directory>/p_to_t.png

    Args:
        output_directory: a string denoting the output directory path

        predicted_letter: a one character string representing the original 
        image's category

        target_letter: a one character string representing the tainted 
        image's category

    Returns:
        A string representing the filename of the output image
        
    """
    out_filename = os.path.join(output_directory, 
        '{}_to_{}.png'.format(predicted_letter, target_letter))
    return out_filename

def export_gray_image(gray_np_img, out_filename):
    """
    Saves the NumPy array representing a grayscale image to 
    the given output file

    Args:
        gray_np_img: a 2D NumPy array representing the (normalized) grayscale 
        image

        out_filename: a string denoting the path to save the image to
        

    Returns:
        None
        
    """
    scaled_gray_np_img = 255 * gray_np_img.transpose()
    gray_img = Image.fromarray(scaled_gray_np_img)
    gray_img.convert('RGB').save(out_filename)
    print('Find the tainted image at {}'.format(out_filename))

def main():
    """
    Takes shell inputs, processes the untainted image, produces an adversarial 
    example, and outputs that tainted image.

    Args:
        None

    Returns:
        None

    """
    parser = arg_utils.make_image_adversary_parser()
    args = parser.parse_args()
    
    # Choose config dictionaries
    classifier_config = CLASSIFIER_CONFIGS[args.classifier_architecture]
    adv_config = ADVERSARY_CONFIGS[args.adversary_architecture]

    # Check that the image dimensions are consistent between the configs
    assert classifier_config['img_height'] == adv_config['img_height']
    assert classifier_config['img_width'] == adv_config['img_width']

    orig_input = process_color_image(args.image_filename,
        classifier_config['img_height'], classifier_config['img_width'])
    if args.display_image:
        display_normed_image(orig_input)

    # Update adv_config with the numerical target class
    target_class = ord(args.target_letter) - ord('a')
    adv_config.update({ 'target_class': target_class })

    with tf.Graph().as_default():
        classifier = LetterClassifier(classifier_config)
        adversary = GradAdv(adv_config, classifier, orig_input)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='letter_classifier'))
        with tf.Session() as session:
            session.run(init)
            saver.restore(session, args.checkpoint)
            predicted_class = classifier.eval(session, np.array([orig_input]))
            predicted_letter = chr(ord('a') + predicted_class)
            print('Predicted Class: {}'.format(predicted_class))
            tainted_image = adversary.create_tainted_image(session)
        
    if args.display_image:
        display_normed_image(tainted_image)
    make_output_directory(args.output_directory)
    out_filename = make_output_image_filename(args.output_directory, 
        predicted_letter, args.target_letter)
    export_gray_image(tainted_image, out_filename)

if __name__ == '__main__':
    main()
