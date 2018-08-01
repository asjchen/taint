# Constructs a model for classifying alphabet letters
# Given the image training data, we produce a classifier trained with
# CNN architecture

import random
import numpy as np
import tensorflow as tf

# Note: place the hyperparams in a JSON somewhere else, 
# - img_height
# - img_width
# - num_classes
# - epochs
# - learning_rate
# - batch_size

class LetterClassifier(object):
    def __init__(self, config):
        self.config = config
        self.activation_map = {
            'relu': tf.nn.relu
        }
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, 
            shape=(None, self.config['img_height'], self.config['img_width']))
        self.label_placeholder = tf.placeholder(tf.float32, 
            shape=(None, self.config['num_classes']))

    def create_feed_dict(self, input_batch, label_batch=None):
        feed_dict = { self.input_placeholder: input_batch }
        if label_batch is not None:
            feed_dict.update({ self.label_placeholder: label_batch })
        return feed_dict

    def add_prediction_op(self):
        input_layer = tf.reshape(self.input_placeholder, 
            [-1, self.config['img_height'], self.config['img_width'], 1])

        # Try 1: just a one-layer NN
        # input_flat = tf.reshape(input_layer, 
        #     [-1, self.config['img_height'] * self.config['img_width']])
        # predicted = tf.layers.dense(input_flat, self.config['num_classes'],
        #     activation=self.activation_map[self.config['output_activation']])
        

        # Try 2: basic CNN
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=self.config['conv1_num_filters'],
            kernel_size=self.config['kernel_size'],
            padding='same',
            activation=self.config['activation'])
        pool1 = tf.layers.max_pooling2d(inputs=conv1, 
            pool_size=self.config['pool_size'], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=self.config['conv2_num_filters'],
            kernel_size=self.config['kernel_size'],
            padding='same',
            activation=self.config['activation'])
        pool2 = tf.layers.max_pooling2d(inputs=conv2, 
            pool_size=self.config['pool_size'], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, self.config['pool2_output_dim']])
        dense = tf.layers.dense(inputs=pool2_flat, 
            units=self.config['dense_dim'], activation=self.config['activation'])
        predicted = tf.layers.dense(inputs=dense, units=self.config['num_classes'])
        return predicted

    def add_loss_op(self, predicted):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=predicted, labels=self.label_placeholder)
        loss = tf.reduce_mean(loss)
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config['learning_rate'])
        train_op = optimizer.minimize(loss)
        return train_op

    def measure_performance(self, sess, test_X, test_y):
        feed = self.create_feed_dict(test_X, label_batch=test_y)
        raw_predict, loss = sess.run(
            [self.pred, self.loss], feed_dict=feed)
        # Take the softmax of the raw predictions
        unnormed_pred_probs = np.exp(raw_predict)
        pred_labels = np.argmax(
            unnormed_pred_probs / np.sum(unnormed_pred_probs),
            axis=1)
        actual_labels = np.argmax(test_y, axis=1)
        accuracy = np.sum(actual_labels == pred_labels) / actual_labels.shape[0]
        return loss, accuracy

    def run_epoch(self, sess, train_X, train_y, dev_X, dev_y):
        for idx in range(0, train_X.shape[0], self.config['batch_size']):
            train_X_batch = train_X[idx: idx + self.config['batch_size'], :]
            train_y_batch = train_y[idx: idx + self.config['batch_size'], :]
            feed = self.create_feed_dict(train_X_batch, 
                label_batch=train_y_batch)
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
            curr_log_cnt = idx // self.config['log_per'] 
            prev_log_cnt = (idx - self.config['batch_size']) // self.config['log_per']
            if curr_log_cnt != prev_log_cnt:
                print('Trained on {}/{} with current loss {}'.format(
                    idx, train_X.shape[0], loss))
            

        dev_loss, dev_accuracy = self.measure_performance(sess, dev_X, dev_y)
        print('Dev Loss: {}\nDev Accuracy: {}'.format(dev_loss, dev_accuracy))

    def train(self, sess, train_dev_X, train_dev_y, prop_train=0.8):
        indices = list(range(train_dev_X.shape[0]))
        random.shuffle(indices)
        train_indices = indices[: int(prop_train * len(indices))]
        dev_indices = indices[int(prop_train * len(indices)): ]
        train_X = train_dev_X[train_indices, :]
        train_y = train_dev_y[train_indices]
        dev_X = train_dev_X[dev_indices, :]
        dev_y = train_dev_y[dev_indices, :]
        for i in range(self.config['epochs']):
            print('\nEpoch {} of {}:'.format(i + 1, self.config['epochs']))
            self.run_epoch(sess, train_X, train_y, dev_X, dev_y)

# TODO: evaluate on the test set!
# TODO: Remember to store the model!




