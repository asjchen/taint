# Try 1: basic single dense layer network

single_layer_config = { 
    'img_height': 28, 
    'img_width': 28,
    'num_classes': 26,
    'epochs': 20,
    'learning_rate': 0.001,
    'output_activation': 'relu',
    'batch_size': 2000,
    'log_per': 10000
}

# Try 2: CNN with two convolution+pooling layers, followed by a dense layer
cnn_config = { 
    'img_height': 28, 
    'img_width': 28,
    'num_classes': 26,
    'batch_size': 2000,
    'log_per': 10000,
    'epochs': 20,
    'learning_rate': 0.005, 
    'activation': 'relu',
    'kernel_size': [5, 5],
    'pool_size': [2, 2],
    'conv1_num_filters': 32,
    'conv2_num_filters': 64,
    'pool2_output_dim': 7 * 7 * 64,
    'dense_dim': 1024
}

