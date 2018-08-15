# Gradient Descent Adversarial Attack (white box)

import numpy as np
import tensorflow as tf

class GradAdv(object):
    def __init__(self, config, classifier, orig_image):
        self.config = config
        self.classifier = classifier
        self.orig_image = orig_image

        with tf.variable_scope('taint'):
            self.taint = tf.Variable(tf.random_uniform(
                [self.config['img_height'], self.config['img_width']], 
                dtype=tf.float64))
        self.loss = self.add_loss_op(self.taint)
        self.train_op = self.add_train_op(self.loss)

    def create_feed_dict(self):
        input_batch = np.expand_dims(self.orig_image, 0).astype(np.float64)
        feed_dict = {self.classifier.input_placeholder: input_batch}
        return feed_dict

    def add_loss_op(self, taint):
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
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config['learning_rate'])
        train_op = optimizer.minimize(loss, 
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                    scope='taint'))
        return train_op

    def create_tainted_image(self, sess):
        feed_dict = self.create_feed_dict()
        for i in range(self.config['epochs']):
            print('\nEpoch {} of {}:'.format(i + 1, self.config['epochs']))
            _, loss, taint = sess.run([self.train_op, self.loss, self.taint], feed_dict=feed_dict)
            print('Loss: {}'.format(loss))
        taint, loss = sess.run([self.taint, self.loss], feed_dict=feed_dict)
        taint_norm = np.linalg.norm(taint)
        adv_loss = loss - taint_norm * self.config['norm_constant']
        adv_prob = np.exp(-adv_loss)

        print('\nFinal Norm: {}'.format(taint_norm))
        print('Final Target Probability: {}'.format(adv_prob))
        return np.clip(taint + self.orig_image, 0, 1)





    
