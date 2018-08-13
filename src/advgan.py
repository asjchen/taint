# AdvGAN architecture for producing adversarial noise on top of an input image

import numpy as np
import tensorflow as tf
from letter_classifier import LetterClassifier


class AdvGAN(object):
    def __init__(self, config, classifier, discriminator, generator):
        self.config = config
        self.classifier = classifier
        self.discriminator = discriminator
        self.generator = generator

    def run_epoch(self, sess, train_X, dev_X):
        for idx in range(0, train_X.shape[0], self.config['batch_size']):
            train_X_batch = train_X[idx: idx + self.config['batch_size'], :]

            # feed dictionary
            gen_feed_dict = self.generator.create_feed_dict(train_X_batch)
            taint_batch = self.generator.run(
                [self.generator.predictions], feed_dict=gen_feed_dict)

            disc_feed_dict = self.discriminator.create_feed_dict(
                train_X_batch, taint_batch)

            _, disc_loss = sess.run(
                [self.discriminator.train_op, self.discriminator.loss], 
                feed_dict=disc_feed_dict)

            curr_log_cnt = idx // self.config['log_per'] 
            prev_log_cnt = (idx - self.config['batch_size']) // self.config['log_per']
            
            if curr_log_cnt != prev_log_cnt:
                print('Trained discriminator on {}/{} with current loss {}'.format(
                    idx, train_X.shape[0], disc_loss))

            # train the generator too
            target_batch = np.zeros(
                (train_X_batch.shape[0], self.config['num_classes']))
            target_batch[:, self.config['target_class']] = 1

            gen_feed_dict = self.generator.create_feed_dict(
                train_X_batch)
            _, gen_loss = sess.run(
                [self.generator.train_op, self.generator.loss], 
                feed_dict=gen_feed_dict)

            if curr_log_cnt != prev_log_cnt:
                print('Trained generator on {}/{} with current loss {}'.format(
                    idx, train_X.shape[0], disc_loss))
        # dev -- measure performance on GAN and target accuracies

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


