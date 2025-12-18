# -*- coding: utf-8 -*-
"""
Neural network model definitions for mass spectrometry classification.

This module contains the definition of the VDCNN model architecture.
"""

from tensorflow.keras import layers, models, regularizers

def build_vdcnn_16_multilabel(input_shape, num_classes, kernel_size, 
                             pool_size, strides, dropout_rate):
    """
    Builds a VDCNN (Very Deep Convolutional Neural Network) model for multi-label classification.
    
    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        num_classes (int): Number of output classes.
        kernel_size (int): Size of the convolutional kernels.
        pool_size (int): Size of the max pooling windows.
        strides (int): Stride of the pooling operations.
        dropout_rate (float): Dropout rate for regularization.
        
    Returns:
        tf.keras.Model: Compiled VDCNN model.
    """
    model = models.Sequential()
    
    # Block 1
    model.add(layers.Conv1D(filters=64, kernel_size=kernel_size, 
                           padding='same', activation='relu', 
                           input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=pool_size, strides=strides))
    
    # Block 2
    model.add(layers.Conv1D(filters=128, kernel_size=kernel_size, 
                           padding='same', activation='relu'))
    model.add(layers.Conv1D(filters=128, kernel_size=kernel_size, 
                           padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=pool_size, strides=strides))
    
    # Block 3
    model.add(layers.Conv1D(filters=256, kernel_size=kernel_size, 
                           padding='same', activation='relu'))
    model.add(layers.Conv1D(filters=256, kernel_size=kernel_size, 
                           padding='same', activation='relu'))
    model.add(layers.Conv1D(filters=256, kernel_size=kernel_size, 
                           padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=pool_size, strides=strides))
    
    # Block 4
    model.add(layers.Conv1D(filters=512, kernel_size=kernel_size, 
                           padding='same', activation='relu'))
    model.add(layers.Conv1D(filters=512, kernel_size=kernel_size, 
                           padding='same', activation='relu'))
    model.add(layers.Conv1D(filters=512, kernel_size=kernel_size, 
                           padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=pool_size, strides=strides))
    
    # Block 5
    model.add(layers.Conv1D(filters=512, kernel_size=kernel_size, 
                           padding='same', activation='relu'))
    model.add(layers.Conv1D(filters=512, kernel_size=kernel_size, 
                           padding='same', activation='relu'))
    model.add(layers.Conv1D(filters=512, kernel_size=kernel_size, 
                           padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=pool_size, strides=strides))
    
    # Classification head
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(4096, activation='relu', 
                          kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model