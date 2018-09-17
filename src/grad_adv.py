# Gradient Descent Adversarial Attack (white box)
# Given a letter classifier and an image of a letter, computes a low-norm mask 
# to layer onto the original image (i.e. the original image with a bit of 
# "taint"). This new image should fool the classifier into thinking it's a 
# different letter from the original image.

import numpy as np
import tensorflow as tf

class GradAdv(object):
    """
    Represents an adversarial system that uses gradient descent to minimize
    the loss (-1 * <classifier loss>) + lambda * <norm of taint>^2. This loss 
    maximizes the classifier's probability of the tainted image being the 
    target letter while keeping the "amount of taint" small.

    """

    def __init__(self, config, classifier, orig_image):
        """
        Initializes the GradAdv object with a configuration, classifier, 
        and original image

        Args:
            config: a dictionary containing hyperparameters and settings such
            as img_height, img_width, num_classes (26 for just identifying
            letters), epochs, learning_rate, norm_constant (lambda in the 
            GradAdv description, the weight to place on the taint norm in the 
            loss function), etc. Check hyperparams.py to see the necessary 
            hyperparameters.

            classifier: a LetterClassifier object, representing the trained 
            letter classifier model

            orig_image: a 2D NumPy array of dimensions img_height x img_width 
            that represents the unmodified image to place the taint mask on

        Returns:
            None

        """
        self.config = config
        self.classifier = classifier
        self.orig_image = orig_image

        # Separate this taint variable from the classifier's variables, so 
        # training only affects this variable.
        with tf.variable_scope('taint'):
            self.taint = tf.Variable(tf.random_uniform(
                [self.config['img_height'], self.config['img_width']], 
                dtype=tf.float64))
        self.loss = self.add_loss_op(self.taint)
        self.train_op = self.add_train_op(self.loss)

    def create_feed_dict(self):
        """
        Constructs the feed dictionary for the classifier. In this situation, 
        the input placeholder is meaningless since we're not using it to 
        compute the tainted image's classifier probabilities (opting instead
        to directly use LetterClassifier.compute_prediction())

        Args:
            None

        Returns:
            A dictionary representing the feed dictionary to the classifier's TF 
            graph, starting the computation in that graph

        """
        input_batch = np.expand_dims(self.orig_image, 0).astype(np.float64)
        feed_dict = {self.classifier.input_placeholder: input_batch}
        return feed_dict

    def add_loss_op(self, taint):
        """
        Given the mask/taint, computes the loss for the adversarial system.
        The loss is (-1 * <classifier loss>) + lambda * <norm of taint>^2,
        where the classifier loss is the classifier's probability of the 
        tainted image representing the target letter (i.e. how strongly 
        the adversary fooled the classifier). Here, we use L2 norm for the 
        taint, and the hyperparameter lambda is in self.config as 
        'norm_constant'

        Args:
            taint: a 2D tensor of dimensions img_height x img_width

        Returns:
            A float tensor/operation representing the loss 

        """
        taint_norm = tf.norm(taint)
        new_image = tf.clip_by_value(taint + self.orig_image, 0, 1)
        classifier_inputs = tf.expand_dims(new_image, 0)
        classifier_pred = self.classifier.compute_prediction(classifier_inputs)
        adv_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=classifier_pred, 
            labels=tf.one_hot([self.config['target_class']], 
                self.config['num_classes']))
        adv_loss = tf.reduce_mean(adv_loss)
        total_loss = adv_loss + self.config['norm_constant'] * (taint_norm ** 2)
        return total_loss

    def add_train_op(self, loss):
        """
        Updates the appropriate variables based on the current loss, as per
        gradient descent

        Args:
            loss: float tensor representing the current loss, from 
            self.add_loss_op()

        Returns:
            An operation that updates the necessary variables
        
        """
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config['learning_rate'])
        train_op = optimizer.minimize(loss, 
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                scope='taint'))
        return train_op

    def create_tainted_image(self, sess):
        """
        Creates the tainted/modified image that the classifier (should) 
        incorrectly identify as the target_letter (from self.config).

        Args:
            sess: a Tensorflow session

        Returns:
            A 2D NumPy array of dimensions img_height x img_width, 
            representing the tainted image (the original image, layered with 
            taint). All the values are in [0, 1].
        
        """
        feed_dict = self.create_feed_dict()
        for i in range(self.config['epochs']):
            print('\nEpoch {} of {}:'.format(i + 1, self.config['epochs']))
            _, loss, taint = sess.run([self.train_op, self.loss, self.taint], 
                feed_dict=feed_dict)
            print('Loss: {}'.format(loss))
        taint, loss = sess.run([self.taint, self.loss], feed_dict=feed_dict)
        taint_norm = np.linalg.norm(taint)
        adv_loss = loss - (taint_norm ** 2) * self.config['norm_constant']
        adv_prob = np.exp(-adv_loss)

        print('\nFinal Norm: {}'.format(taint_norm))
        print('Final Target Probability: {}'.format(adv_prob))
        return np.clip(taint + self.orig_image, 0, 1)
    
