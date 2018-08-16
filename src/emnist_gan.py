# Trains the GAN on part of the EMNIST dataset

import argparse
import numpy as np
import tensorflow as tf
from letter_classifier import LetterClassifier
from emnist import emnist_csv_to_xy
from hyperparams import CLASSIFIER_CONFIGS, ADVERSARY_CONFIGS
# from adv_discriminator import AdvDiscriminator
# from adv_generator import AdvGenerator
from advgan import AdvGAN

def main():
    parser = argparse.ArgumentParser(description=('Trains the GAN model on '
        'a subset of the EMNIST dataset'))
    parser.add_argument('train_file', type=argparse.FileType('r'),
        help='File for the EMNIST training dataset')
    parser.add_argument('-a', '--classifier_architecture', 
        choices=CLASSIFIER_CONFIGS.keys(), default='cnn_two_layer',
        help=('Classifier architecture to be used, one of {}, default '
            'is cnn_two_layer'.format(list(CLASSIFIER_CONFIGS.keys()))))
    parser.add_argument('-g', '--gan_architecture', 
        choices=ADVERSARY_CONFIGS.keys(), default='advgan',
        help=('GAN architecture to be used, one of {}, default '
            'is advgan'.format(list(ADVERSARY_CONFIGS.keys()))))
    parser.add_argument('-c', '--checkpoint', default='tmp/model.ckpt',
        help=('File with classifier model checkpoint if the model has '
            'already been trained'))
    parser.add_argument('-t', '--target_letter', default='m',
        help='Lowercase letter that serves as target class')
    args = parser.parse_args()

    target_class = ord(args.target_letter) - ord('a')
    train_X, train_y = emnist_csv_to_xy(args.train_file)

    classifier_config = CLASSIFIER_CONFIGS[args.classifier_architecture]
    gan_config = ADVERSARY_CONFIGS[args.gan_architecture]
    gan_config.update({ 'target_class': target_class })

    with tf.Graph().as_default():
        classifier = LetterClassifier(classifier_config)
        # discriminator = AdvDiscriminator(gan_config)
        # generator = AdvGenerator(gan_config, classifier, discriminator)
        # gan = AdvGAN(gan_config, classifier, discriminator, generator)
        gan = AdvGAN(gan_config, classifier)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='letter_classifier'))
        with tf.Session() as session:
            session.run(init)
            saver.restore(session, args.checkpoint)
            gan.train(session, train_X)

if __name__ == '__main__':
    main()
