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

    def residual_layer(self, x, num_filters):
        first_layer = tf.contrib.layers.conv2d(x, num_filters,
            kernel_size=[3, 3],
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu)
        second_layer = tf.contrib.layers.conv2d(first_layer, num_filters,
            kernel_size=[3, 3],
            stride=1,
            padding='SAME')
        return second_layer + x

    def add_prediction_op(self):
        # TODO: do I need a noise placeholder?
        # c3s1-8, d16, d32, r32, r32, r32, r32, u16, u8, c3s1-3

        inputs = tf.reshape(self.input_placeholder, 
            [-1, self.config['img_height'], self.config['img_width'], 1])
        conv_layer1 = tf.contrib.layers.conv2d(inputs, 8,
            kernel_size=[3, 3],
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu)
        conv_layer2 = tf.contrib.layers.conv2d(conv_layer1, 16,
            kernel_size=[3, 3],
            stride=2,
            padding='SAME',
            activation_fn=tf.nn.relu)
        conv_layer3 = tf.contrib.layers.conv2d(conv_layer2, 32,
            kernel_size=[3, 3],
            stride=2,
            padding='SAME',
            activation_fn=tf.nn.relu)

        res_layer1 = self.residual_layer(conv_layer3, 32)
        res_layer2 = self.residual_layer(res_layer1, 32)
        res_layer3 = self.residual_layer(res_layer2, 32)
        res_layer4 = self.residual_layer(res_layer3, 32)

        conv_layer4 = tf.contrib.layers.conv2d_transpose(res_layer4, 16,
            kernel_size=[3, 3],
            stride=2,
            padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.contrib.layers.instance_norm)
        conv_layer5 = tf.contrib.layers.conv2d_transpose(conv_layer4, 8,
            kernel_size=[3, 3],
            stride=2,
            padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.contrib.layers.instance_norm)
        conv_layer6 = tf.contrib.layers.conv2d(conv_layer5, 3,
            kernel_size=[3, 3],
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu)
        return conv_layer6



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




