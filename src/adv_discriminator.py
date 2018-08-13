# AdvGAN discriminator class

import numpy as np
import tensorflow as tf
from letter_classifier import LetterClassifier

class AdvDiscriminator(object):
    def __init__(self, config):
        self.config = config
        self.add_placeholders()
        self.predicted_orig = self.add_prediction_op()
        self.predicted_gen = self.add_prediction_op(add_mask=True)
        self.loss = self.add_loss_op(self.predicted_orig, self.predicted_gen)
        self.train_op = self.add_training_op(self.loss)

    # add placeholders
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float64, 
            shape=(None, self.config['img_height'], self.config['img_width']))
        self.taint_placeholder = tf.placeholder(tf.float64, 
            shape=(None, self.config['img_height'], self.config['img_width']))

    def create_feed_dict(self, input_batch, taint_batch):
        feed_dict = { 
            self.input_placeholder: input_batch,
            self.taint_placeholder: taint_batch
        }
        return feed_dict

    def add_prediction_op(self, add_mask=False):
        inputs = self.input_placeholder
        if add_mask:
            inputs += self.taint_placeholder

        # C8, C16, C32, FC
        inputs = tf.reshape(inputs, 
            [-1, self.config['img_height'], 
            self.config['img_width'], 1])
        conv_layer1 = tf.contrib.layers.conv2d(inputs, 8,
            kernel_size=[4, 4],
            stride=2,
            padding='SAME',
            activation_fn=tf.nn.leaky_relu)

        conv_layer2 = tf.contrib.layers.conv2d(conv_layer1, 16,
            kernel_size=[4, 4],
            stride=2,
            padding='SAME',
            activation_fn=tf.nn.leaky_relu, # default slope is 0.2
            normalizer_fn=tf.contrib.layers.instance_norm)

        conv_layer3 = tf.contrib.layers.conv2d(conv_layer1, 32,
            kernel_size=[4, 4],
            stride=2,
            padding='SAME',
            activation_fn=tf.nn.leaky_relu, # default slope is 0.2
            normalizer_fn=tf.contrib.layers.instance_norm)

        raw_predictions = tf.layers.dense(tf.contrib.layers.flatten(conv_layer3), 
            self.config['num_classes'], activation=tf.nn.leaky_relu)
        both_predictions = tf.nn.softmax(raw_predictions)
        predictions = tf.slice(both_predictions, [0, 0], [-1, 1])
        return predictions

    # Here, predicted_orig and predicted_gen are just 1D rows of probabilities
    def add_loss_op(self, predicted_orig, predicted_gen):
        # only care about the GAN loss for this one
        loss = tf.log(predicted_orig) + tf.log(1 - predicted_gen)
        loss = -1 * tf.reduce_mean(loss) 
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config['learning_rate'])
        # Disciminator wants to maximize the loss
        train_op = optimizer.minimize(-1 * loss)
        return train_op



#Rather than minimizing log(1- D(G(z))), training the Generator to maximize log D(G(z)) will provide much stronger gradients early in training
