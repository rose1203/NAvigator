# -*- coding: utf-8 -*-
"""
Main training script for VDCNN mass spectrometry classification.

This script orchestrates the model training process and serves as the entry point.
"""

import os
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Import from our modularized files
from models import build_vdcnn_16_multilabel
from utils import load_dataset, data_resample, topK, cal_mean_std

# Set environment for potential OpenMP issues
os.environ['OMP_DISPLAY_ENV'] = 'FALSE'

def train_model(total_df, fn_groups, kernel_size, pool_size, strides,
               dropout_rate, learning_rate, epochs, batch_size, save_path):
    """
    Trains the VDCNN model with comprehensive evaluation.
    
    Args:
        total_df (pd.DataFrame): Dataset for training and validation.
        fn_groups (int): Number of functional groups.
        kernel_size (int): Convolutional kernel size.
        pool_size (int): Max pooling size.
        strides (int): Pooling stride.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        save_path (str): Path to save the trained model.
    """
    # Prepare features and labels
    X = total_df.iloc[:, :-fn_groups]
    y = total_df.iloc[:, -fn_groups:]
    
    # Initialize lists to store metrics across runs
    accuracy_all, recall_all, precision_all, f1_all = [], [], [], []
    precisions_class, recalls_class, f1_scores_class = [], [], []
    
    class_labels = y.columns.tolist()
    print(f"Training model for classes: {class_labels}")
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, 
                                  restore_best_weights=True)
    
    # 10-fold cross-validation style evaluation
    for i in range(10):
        print(f"\n--- Cross-validation fold {i+1}/10 ---")
        
        # Split data with stratification
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, 
                                                         stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                           test_size=0.11, 
                                                           stratify=y_train)
        
        # Balance training data
        df_train = pd.concat([X_train, y_train], axis=1)
        df_train = data_resample(fn_groups, df_train)
        
        # Prepare data for model
        X_train = df_train.iloc[:, :-fn_groups]
        y_train = df_train.iloc[:, -fn_groups:]
        
        # Add channel dimension for Conv1D
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        
        input_shape = X_train.shape[1:]
        num_classes = y_train.shape[1]
        
        # Build and compile model
        model = build_vdcnn_16_multilabel(input_shape, num_classes, kernel_size,
                                       pool_size, strides, dropout_rate)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        # Train model
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                 epochs=epochs, batch_size=batch_size, verbose=2,
                 callbacks=[early_stopping])
        
        # Freeze model for saving
        model.trainable = False
        tf.saved_model.save(model, save_path)
        
        # Generate predictions and calculate metrics
        y_val_pred = model.predict(X_val)
        y_val = np.array(y_val, dtype=np.int64)
        y_pred_processed = topK(y_val, y_val_pred, k=3)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred_processed)
        recall = recall_score(y_val, y_pred_processed, average='weighted')
        precision = precision_score(y_val, y_pred_processed, average='weighted')
        f1 = f1_score(y_val, y_pred_processed, average='weighted')
        
        # Store rounded metrics
        for metric, value in [('Accuracy', accuracy), ('Precision', precision),
                            ('Recall', recall), ('F1', f1)]:
            rounded_value = round(value * 100, 5)
            eval(metric.lower() + '_all').append(rounded_value)
            print(f'Validation {metric} {i}: {rounded_value}')
        
        # Store per-class metrics for detailed analysis
        precision_per_class = precision_score(y_val, y_pred_processed, 
                                            average=None, labels=np.arange(len(class_labels)))
        recall_per_class = recall_score(y_val, y_pred_processed, 
                                     average=None, labels=np.arange(len(class_labels)))
        f1_per_class = f1_score(y_val, y_pred_processed, average=None)
        
        precisions_class.append(precision_per_class)
        recalls_class.append(recall_per_class)
        f1_scores_class.append(f1_per_class)
    
    # Print hyperparameters for reproducibility
    print(f"\nHyperparameters: kernel_size={kernel_size}, pool_size={pool_size}, "
          f"strides={strides}, learning_rate={learning_rate}, epochs={epochs}, "
          f"batch_size={batch_size}")
    
    # Print overall statistics
    for metric_name, metric_list in [('Accuracy', accuracy_all), 
                                    ('Precision', precision_all),
                                    ('Recall', recall_all), 
                                    ('F1', f1_all)]:
        cal_mean_std(metric_name, metric_list)

def main():
    """
    Main function that orchestrates the training process.
    """
    print("Starting VDCNN training for mass spectrometry data...")
    
    # Load dataset
    total_df, fn_groups = load_dataset('./data/target_pipeline_1A3.csv')
    
    # Train model
    train_model(total_df, fn_groups, kernel_size=5, pool_size=2, strides=2,
                dropout_rate=0.5, learning_rate=0.00005, epochs=200, 
                batch_size=64, save_path="model/saved_model_t/pipeline_1A3")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()