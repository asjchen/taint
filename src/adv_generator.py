# AdvGAN generator class

import numpy as np
import tensorflow as tf
from letter_classifier import LetterClassifier

class AdvGenerator(object):
    def __init__(self, config, classifier, discriminator):
        self.config = config
        self.classifier = classifier
        self.discriminator = discriminator
        self.add_placeholders()

        self.predicted_mask = self.add_prediction_op()
        self.loss = self.add_loss_op(self.predicted_mask)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float64, 
            shape=(None, self.config['img_height'], self.config['img_width']))

    def create_feed_dict(self, input_batch, target_batch=None):
        feed_dict = { 
            self.input_placeholder: input_batch,
            self.classifier.input_placeholder: input_batch
        }
        if target_batch is not None:
            feed_dict.update(
                { self.classifier.label_placeholder: target_batch })
        return feed_dict

    def add_prediction_op(self):
        # TODO: do I need a noise placeholder?
        # c3s1-8, d16, d32, r32, r32, r32, r32, u16, u8, c3s1-3

        #conv_layer1 = 




        return self.input_placeholder * 0

    def add_loss_op(self, predicted):
        disc_orig = self.discriminator.predicted_orig
        disc_gen = self.discriminator.predicted_gen
        gan_loss = tf.log(disc_orig) + tf.log(1 - disc_gen)
        gan_loss = tf.reduce_mean(gan_loss)

        adv_loss = self.classifier.loss

        gen_norm = tf.norm(predicted, axis=1)
        hinge_loss = tf.maximum(tf.cast(0.0, dtype=tf.float64), 
            gen_norm - self.config['noise_bound'])
        hinge_loss = tf.reduce_mean(hinge_loss)

        total_loss = adv_loss
        total_loss += self.config['gan_constant'] * gan_loss
        total_loss += self.config['hinge_constant'] * hinge_loss
        return total_loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config['learning_rate'])
        train_op = optimizer.minimize(loss)
        return train_op




