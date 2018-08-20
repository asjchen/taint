# Constructs a model for classifying alphabet letters
# Given the image training data, we produce a classifier trained with
# deep learning architecture

import random
import numpy as np
import tensorflow as tf

class LetterClassifier(object):
    """
    Represents a classifier for identifying a letter of the alphabet given
    a grayscale image, a 2D array/tensor with values in [0, 1]. Used to 
    train and evaluate a classifier model.

    """

    def __init__(self, config):
        """
        Initializes the LetterClassifier object with a configuration

        Args:
            config: a dictionary containing hyperparameters and settings such
            as img_height, img_width, num_classes (26 for just identifying
            letters), epochs, learning_rate, batch_size, etc.

        Returns:
            None

        """
        self.config = config
        self.activation_map = {
            'relu': tf.nn.relu
        }
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        """
        Adds the placeholders (inputs and one-hot label vectors) to the TF
        graph, as these change from batch to batch. Note that the inputs have
        dimension [None, img_height, img_width] and the label vectors have
        dimension [None, num_classes].

        Args:
            None

        Returns:
            None

        """
        self.input_placeholder = tf.placeholder(tf.float64, 
            shape=(None, self.config['img_height'], self.config['img_width']))
        self.label_placeholder = tf.placeholder(tf.float64, 
            shape=(None, self.config['num_classes']))

    def create_feed_dict(self, input_batch, label_batch=None):
        """
        Constructs the feed dictionary, which consists of the input batch as
        well as the label vectors (in training and evaluation).

        Args:
            input_batch: a 3D array of dimensions [None, img_height, img_width]
            that represents part of the data inputs. The first dimension
            is often the batch_size.

            label_batch: a 2D array of dimensions [None, num_classes] that
            represents part of the data labels. The first dimension is often
            the batch_size.

        Returns:
            A dictionary representing the feed dictionary to the TF graph,
            starting the computation in the graph

        """
        feed_dict = { self.input_placeholder: input_batch }
        if label_batch is not None:
            feed_dict.update({ self.label_placeholder: label_batch })
        return feed_dict

    def compute_prediction(self, inputs):
        input_layer = tf.reshape(inputs, 
            [-1, self.config['img_height'], self.config['img_width'], 1])

        with tf.variable_scope('letter_classifier', reuse=tf.AUTO_REUSE):
            if self.config['name'] == 'single_layer':
                # Try 1: just a one-layer NN
                input_flat = tf.reshape(input_layer, 
                    [-1, self.config['img_height'] * self.config['img_width']])
                predicted = tf.layers.dense(input_flat, self.config['num_classes'],
                    activation=self.activation_map[self.config['output_activation']])
            
            elif self.config['name'] == 'cnn_two_layer':
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
            
            else:
                raise Exception('Wrong config name for classifier.')
            
        return predicted

    def add_prediction_op(self):
        """
        Given the input placeholder, produces the tensor representing the 
        model's predictions of the inputs' classes. The architectures
        supported are the names in CLASSIFIER_CONFIGS in hyperparams.py

        Args:
            None

        Returns:
            A tensor/operation representing the logged probabilities for 
            each class

        """
        return self.compute_prediction(self.input_placeholder)
        

    def add_loss_op(self, predicted):
        """
        Given the predicted log probabilities (and the labels placeholder),
        calculates the average cross entropy loss.

        Args:
            predicted: tensor representing the logged probabilities for each 
            class, from self.add_prediction_op()

        Returns:
            A float tensor/operation representing the loss (average over the 
            batch)

        """
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=predicted, labels=self.label_placeholder)
        loss = tf.reduce_mean(loss)
        return loss

    def add_training_op(self, loss):
        """
        Updates the appropriate variables based on the current loss, as per
        gradient descent

        Args:
            loss: float tensor representing the loss over the current batch, 
            from self.add_loss_op()

        Returns:
            An operation that updates the necessary variables
        
        """
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config['learning_rate'])
        train_op = optimizer.minimize(loss)
        return train_op

    def measure_performance(self, sess, test_X, test_y):
        """
        Evaluates the model's performance on a labeled test/dev set

        Args:
            sess: a Tensorflow session

            test_X: a NumPy array of dimensions [None, img_height, img_width]
            representing the inputs of the test set

            test_y: a NumPy array of dimensions [None, num_classes] consisting
            of the one-hot label vectors of the test set

        Returns:
            An operation that updates the necessary variables
        
        """
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
        """
        Runs one epoch of training, which consists of iterating through
        the entire training set in batches, predicting those batches' labels,
        and updating the variables accordingly.

        Args:
            sess: a Tensorflow session

            train_X: a NumPy array of dimensions [None, img_height, img_width]
            representing the inputs of the training set

            train_y: a NumPy array of dimensions [None, num_classes] consisting
            of the one-hot label vectors of the training set

            dev_X: a NumPy array of dimensions [None, img_height, img_width]
            representing the inputs of the dev set

            dev_y: a NumPy array of dimensions [None, num_classes] consisting
            of the one-hot label vectors of the dev set

        Returns:
            None
        
        """
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
        """
        Trains the model by running several epochs of training (calling 
        run_epoch)

        Args:
            sess: a Tensorflow session

            train_dev_X: a NumPy array of dimensions [None, img_height, 
            img_width] representing the inputs of the combined training and 
            dev set

            train_dev_y: a NumPy array of dimensions [None, num_classes] 
            consisting of the one-hot label vectors of the combined training 
            and dev set

            prop_train: the proportion of the combined training and dev data
            that should be part of the training set (by default, the model 
            trains on the first 80% of the data and evaluates on the remaining
            20% of the data).

        Returns:
            None
        
        """
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

    def eval(self, sess, unknown_X):
        """
        Predicts the classes for a new test set, whose labels are currently 
        unknown

        Args:
            sess: a Tensorflow session

            unknown_X: a NumPy array of dimensions [None, img_height, 
            img_width] representing the inputs of the test set

        Returns:
            A NumPy (integer) array of dimensions [None], representing the 
            classes of the rows in unknown_X
        
        """
        feed = self.create_feed_dict(unknown_X)
        raw_predict = sess.run([self.pred], feed_dict=feed)[0]
        pred_classes = np.argmax(raw_predict, axis=1)
        return pred_classes
        
