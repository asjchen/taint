# Hyperparameter configurations for the classifier and the adversary
# For information about the classifier architectures, 
# check add_prediction_op() in letter_classifier.LetterClassifier

CLASSIFIER_CONFIGS = {
    # Try 1: basic single dense layer network
    'single_layer': { 
        'name': 'single_layer',
        'img_height': 28, # height of the training images
        'img_width': 28, # width of the training images
        'num_classes': 26, # number of possible output (letter) classes 
        'epochs': 20, # number of training epochs
        'learning_rate': 0.001, # learning rate of the Adam optimizer
        'output_activation': 'relu', # activation function for the single layer
        'batch_size': 1000, # number of data samples per batch/update
        'log_per': 5000 # report the current loss after every n samples
    },
    # Try 2: CNN with two convolution+pooling layers, followed by a dense layer
    'cnn_two_layer': { 
        'name': 'cnn_two_layer',
        'img_height': 28, # height of the training images
        'img_width': 28, # width of the training images
        'num_classes': 26, # number of possible output (letter) classes 
        'batch_size': 1000, # number of data samples per batch/update
        'log_per': 5000, # report the current loss after every n samples
        'epochs': 10, # number of training epochs
        'learning_rate': 0.005, # learning rate of the Adam optimizer
        'activation': 'relu', # activation function for all applicable layers
        'kernel_size': [5, 5], # kernel size for convolution layers
        'pool_size': [2, 2], # pooling size for pooling layers
        'conv1_num_filters': 32, # filters for the first convolution
        'conv2_num_filters': 64, # filters for the second convolution
        'pool2_output_dim': 7 * 7 * 64, # output dim after 2nd convolution
        'dense_dim': 1024 # dimension of the final hidden (FC) layer
    }
}

ADVERSARY_CONFIGS = {
    'gradient_descent': {
        'name': 'gradient_descent',
        'img_height': 28, # height of the training images
        'img_width': 28, # width of the training images
        'num_classes': 26, # number of possible output (letter) classes 
        'epochs': 300, # number of training epochs
        'learning_rate': 0.05, # learning rate of the Adam optimizer
        'norm_constant': 0.1 # multiplier for the taint norm in the loss
    }
}
