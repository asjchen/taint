# Processes a raw image to resemble the emnist format
# Then, identifies the class of the resulting image
# Finally, runs an adversary to modify the image for a targerted attack

import argparse
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
    scaled_np = 255.0 * normed_np
    plt.imshow(scaled_np.transpose())
    plt.show()
    plt.close()

def export_gray_image(gray_np_img, orig_letter, new_letter, out_filename):
    scaled_gray_np_img = 256 * gray_np_img.transpose()
    gray_img = Image.fromarray(scaled_gray_np_img)
    gray_img.convert('RGB').save(out_filename)

def main():
    # TODO: consolidate the arguments
    parser = argparse.ArgumentParser(description=('Takes an image and a '
        'classifier model and outputs an adversarial example'))
    parser.add_argument('image_filename', 
        help=('Filename of the (color) input image; the letter should be '
            'white on black background'))
    parser.add_argument('-d', '--display_image', action='store_true',
        help='Displays the grayscale scaled image')
    parser.add_argument('-c', '--classifier_architecture', 
        choices=CLASSIFIER_CONFIGS.keys(), default='cnn_two_layer',
        help=('Classifier architecture to be used, one of {}, default '
            'is cnn_two_layer'.format(list(CLASSIFIER_CONFIGS.keys()))))
    parser.add_argument('-a', '--adversary_architecture', 
        choices=ADVERSARY_CONFIGS.keys(), default='gradient_descent',
        help=('Adversary architecture to be used, one of {}, default '
            'is gradient_descent'.format(list(ADVERSARY_CONFIGS.keys()))))
    parser.add_argument('-p', '--checkpoint', default='tmp/model.ckpt',
        help=('File with classifier model checkpoint if the model has '
            'already been trained'))
    parser.add_argument('-t', '--target_letter', default='m',
        help='Lowercase letter that serves as target class')
    parser.add_argument('-o', '--output_directory', default='bin',
        help='Directory to place tainted image in')
    args = parser.parse_args()
    
    # Original input for the classifier, not for this program
    orig_input = process_color_image(args.image_filename)
    if args.display_image:
        display_normed_image(orig_input)

    classifier_config = CLASSIFIER_CONFIGS[args.classifier_architecture]
    adv_config = ADVERSARY_CONFIGS['gradient_descent']

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
        
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)
        out_filename = os.path.join(args.output_directory, 
            '{}_to_{}.png'.format(predicted_letter, args.target_letter))
        export_gray_image(tainted_image, predicted_letter, 
            args.target_letter, out_filename)



if __name__ == '__main__':
    main()
