# AdvGAN architecture for producing adversarial noise on top of an input image

import numpy as np
import tensorflow as tf
from letter_classifier import LetterClassifier
import random

class AdvGAN(object):
    def __init__(self, config, classifier, discriminator, generator):
        self.config = config
        self.classifier = classifier
        self.discriminator = discriminator
        self.generator = generator

    def measure_performance(self, sess, test_X):
        targets = np.zeros(
                (test_X.shape[0], self.config['num_classes']))
        targets[:, self.config['target_class']] = 1.0

        gen_feed_dict = self.generator.create_feed_dict(test_X,
            target_batch=targets)

        taint = sess.run([self.generator.predicted_mask], 
            feed_dict=gen_feed_dict)[0]

        disc_feed_dict = self.discriminator.create_feed_dict(
            test_X, taint)
        pred_orig, pred_gen = sess.run(
            [self.discriminator.predicted_orig, self.discriminator.predicted_gen], 
            feed_dict=disc_feed_dict)

        classifier_feed_dict = self.classifier.create_feed_dict(
            test_X + taint, label_batch=targets)
        classifier_pred, classifier_loss = sess.run(
            [self.classifier.pred, self.classifier.loss], 
            feed_dict=classifier_feed_dict)

        gen_feed_dict = self.generator.create_feed_dict(test_X,
            target_batch=targets, disc_orig=pred_orig, disc_gen=pred_gen,
            classifier_loss=classifier_loss)
        loss, taint = sess.run(
            [self.generator.loss, self.generator.predicted_mask], 
            feed_dict=gen_feed_dict)

        num_target_correct = np.sum(np.argmax(classifier_pred, axis=1) == self.config['target_class'])
        target_accuracy = num_target_correct / test_X.shape[0]

        orig_gan_accuracy = np.sum(np.argmax(pred_orig, axis=1) == 1) / test_X.shape[0]
        gen_gan_accuracy = np.sum(np.argmax(pred_gen, axis=1) == 0) / test_X.shape[0]

        taint_l2_mag = np.max(np.linalg.norm(taint, axis=1))
        taint_max = np.max(np.linalg.norm(taint, ord=np.inf, axis=1))

        print('Accuracy for fooling the classifier: {}'.format(target_accuracy))
        print('Accuracy for identifying untainted images: {}'.format(orig_gan_accuracy))
        print('Accuracy for identifying tainted images: {}'.format(gen_gan_accuracy))
        print('Maximum L2 for taint: {}'.format(taint_l2_mag))
        print('Maximum L-inf for taint: {}'.format(taint_max))


    def run_epoch(self, sess, train_X, dev_X):
        for idx in range(0, train_X.shape[0], self.config['batch_size']):
            train_X_batch = train_X[idx: idx + self.config['batch_size'], :]
            target_batch = np.zeros(
                (train_X_batch.shape[0], self.config['num_classes']))
            target_batch[:, self.config['target_class']] = 1.0

            # feed dictionary
            gen_feed_dict = self.generator.create_feed_dict(train_X_batch,
                target_batch=target_batch)
            taint_batch = sess.run(
                [self.generator.predicted_mask], feed_dict=gen_feed_dict)[0]
            disc_feed_dict = self.discriminator.create_feed_dict(
                train_X_batch, taint_batch)

            _, disc_orig, disc_gen, disc_loss = sess.run(
                [self.discriminator.train_op, self.discriminator.predicted_orig, 
                self.discriminator.predicted_gen, self.discriminator.loss], 
                feed_dict=disc_feed_dict)

            classifier_feed_dict = self.classifier.create_feed_dict(
                train_X_batch + taint_batch, label_batch=target_batch)
            classifier_loss = sess.run([self.classifier.loss], 
                feed_dict=classifier_feed_dict)[0]

            gen_feed_dict = self.generator.create_feed_dict(
                train_X_batch, target_batch=target_batch, disc_orig=disc_orig,
                disc_gen=disc_gen, classifier_loss=classifier_loss)

            gen_loss = sess.run(
                [self.generator.loss], 
                feed_dict=gen_feed_dict)[0]

            curr_log_cnt = idx // self.config['log_per'] 
            prev_log_cnt = (idx - self.config['batch_size']) // self.config['log_per']
            
            if curr_log_cnt != prev_log_cnt:
                print('Trained discriminator on {}/{} with current loss {}'.format(
                    idx, train_X.shape[0], gen_loss))

            # train the generator too
            gen_feed_dict = self.generator.create_feed_dict(
                train_X_batch, target_batch=target_batch)

            taint_batch = sess.run([self.generator.predicted_mask], 
                feed_dict=gen_feed_dict)[0]

            disc_feed_dict = self.discriminator.create_feed_dict(
                train_X_batch, taint_batch)

            disc_orig, disc_gen = sess.run(
                [self.discriminator.predicted_orig, self.discriminator.predicted_gen], 
                feed_dict=disc_feed_dict)

            classifier_feed_dict = self.classifier.create_feed_dict(
                train_X_batch + taint_batch, label_batch=target_batch)
            classifier_loss = sess.run([self.classifier.loss], 
                feed_dict=classifier_feed_dict)[0]

            gen_feed_dict = self.generator.create_feed_dict(
                train_X_batch, target_batch=target_batch,
                disc_orig=disc_orig, disc_gen=disc_gen, classifier_loss=classifier_loss)

            _, gen_loss = sess.run(
                [self.generator.train_op, self.generator.loss], 
                feed_dict=gen_feed_dict)


            if curr_log_cnt != prev_log_cnt:
                print('Trained generator on {}/{} with current loss {}'.format(
                    idx, train_X.shape[0], gen_loss))
        self.measure_performance(sess, dev_X)
        
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






