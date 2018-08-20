# Hyperparameter configurations for the classifier and the adversary
# For information about the classifier architectures, 
# check add_prediction_op() in letter_classifier.LetterClassifier

CLASSIFIER_CONFIGS = {
    # Try 1: basic single dense layer network
    'single_layer': { 
        'name': 'single_layer',
        'img_height': 28, 
        'img_width': 28,
        'num_classes': 26,
        'epochs': 20,
        'learning_rate': 0.001,
        'output_activation': 'relu',
        'batch_size': 1000,
        'log_per': 5000
    },
    # Try 2: CNN with two convolution+pooling layers, followed by a dense layer
    'cnn_two_layer': { 
        'name': 'cnn_two_layer',
        'img_height': 28, 
        'img_width': 28,
        'num_classes': 26,
        'batch_size': 1000,
        'log_per': 5000,
        'epochs': 10, 
        'learning_rate': 0.005, 
        'activation': 'relu',
        'kernel_size': [5, 5],
        'pool_size': [2, 2],
        'conv1_num_filters': 32,
        'conv2_num_filters': 64,
        'pool2_output_dim': 7 * 7 * 64,
        'dense_dim': 1024
    }
}

ADVERSARY_CONFIGS = {
    'gradient_descent': {
        'name': 'gradient_descent',
        'img_height': 28, 
        'img_width': 28,
        'num_classes': 26,
        'epochs': 300,
        'learning_rate': 0.05,
        'norm_constant': 0.04
    }
}
