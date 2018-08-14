# AdvGAN architecture for producing adversarial noise on top of an input image

import numpy as np
import tensorflow as tf
from letter_classifier import LetterClassifier
import random

class AdvGAN(object):
    def __init__(self, config, classifier):
        self.config = config
        self.classifier = classifier
        self.add_placeholders()
        self.taint = self.add_taint_op()
        self.disc_orig = self.add_disc_op(self.taint, use_taint=False)
        self.disc_gen = self.add_disc_op(self.taint, use_taint=True)
        self.loss = self.add_loss_op(self.taint, self.disc_orig, self.disc_gen)
        self.disc_train_op = self.add_training_op(self.loss, minimize=False)
        self.gen_train_op = self.add_training_op(self.loss, minimize=True)

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float64, 
            shape=(None, self.config['img_height'], self.config['img_width']))

    def create_feed_dict(self, input_batch, target_batch):
        feed_dict = { 
            self.input_placeholder: input_batch,
            self.target_placeholder: target_batch
        }
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

    def add_taint_op(self):
        inputs = tf.reshape(self.input_placeholder, 
            [-1, self.config['img_height'], self.config['img_width'], 1])
        
        with tf.variable_scope('generator'):
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
            conv_layer6 = tf.contrib.layers.conv2d(conv_layer5, 1,
                kernel_size=[3, 3],
                stride=1,
                padding='SAME',
                activation_fn=tf.nn.relu)
        return tf.reshape(conv_layer6, 
            [-1, self.config['img_height'], self.config['img_width']])


    def add_disc_op(self, taint, use_taint=True):
        inputs = self.input_placeholder
        if use_taint:
            inputs += taint

        with tf.variable_scope('discriminator'):
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

    def add_loss_op(self, taint, disc_orig, disc_gen):
        gan_loss = tf.log(disc_orig) + tf.log(1 - disc_gen)
        gan_loss = tf.reduce_mean(gan_loss)

        classifier_pred = self.classifier.compute_prediction(self, inputs)
       
        adv_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=classifier_pred, labels=self.target_placeholder)

        gen_norm = tf.norm(predicted, axis=1)
        hinge_loss = tf.maximum(tf.cast(0.0, dtype=tf.float64), 
            gen_norm - self.config['noise_bound'])
        hinge_loss = tf.reduce_mean(hinge_loss)

        total_loss = adv_loss
        total_loss += self.config['gan_constant'] * gan_loss
        total_loss += self.config['hinge_constant'] * hinge_loss
        return total_loss

    def add_training_op(self, loss, minimize=True):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config['learning_rate'])

        # use the appropriate scope of variables
        if minimize:
            train_op = optimizer.minimize(loss, 
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                    scope='generator'))
        else:
            train_op = optimizer.minimize(-1 * loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                    scope='discriminator'))
        return train_op

    def run_epoch(self, sess, train_X, dev_X):
        for idx in range(0, train_X.shape[0], self.config['batch_size']):
            train_X_batch = train_X[idx: idx + self.config['batch_size'], :]
            target_batch = np.zeros(
                (train_X_batch.shape[0], self.config['num_classes']))
            target_batch[:, self.config['target_class']] = 1.0

            feed_dict = self.create_feed_dict(train_X_batch, target_batch)
            
            _, loss = sess.run([self.disc_train_op, self.loss], 
                feed_dict=feed_dict)

            curr_log_cnt = idx // self.config['log_per'] 
            prev_log_cnt = (idx - self.config['batch_size']) // self.config['log_per']
            
            if curr_log_cnt != prev_log_cnt:
                print('Trained discriminator on {}/{} with current loss {}'.format(
                    idx, train_X.shape[0], loss))

            _, loss = sess.run([self.gen_train_op, self.loss], 
                feed_dict=feed_dict)

            curr_log_cnt = idx // self.config['log_per'] 
            prev_log_cnt = (idx - self.config['batch_size']) // self.config['log_per']
            
            if curr_log_cnt != prev_log_cnt:
                print('Trained generator on {}/{} with current loss {}'.format(
                    idx, train_X.shape[0], loss))

    def train(self, sess, train_dev_X, prop_train=0.8):
        indices = list(range(train_dev_X.shape[0]))
        random.shuffle(indices)
        train_indices = indices[: int(prop_train * len(indices))]
        dev_indices = indices[int(prop_train * len(indices)): ]
        train_X = train_dev_X[train_indices, :]
        dev_X = train_dev_X[dev_indices, :]
        for i in range(self.config['epochs']):
            print('\nEpoch {} of {}:'.format(i + 1, self.config['epochs']))
            self.run_epoch(sess, train_X, dev_X)


