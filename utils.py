# -*- coding: utf-8 -*-
"""
Utility functions for mass spectrometry data processing and evaluation.

This module contains helper functions for data handling, preprocessing, 
and model evaluation metrics.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils import resample

def prepare_target_dataset():
    """
    Prepares the target dataset by aggregating functional groups.
    
    Reads the original target CSV, creates binary indicators for major
    functional groups (1A1, 1A2, 1A3, 2B4), and saves the processed dataset.
    """
    df = pd.read_csv('./data/your_target.csv', index_col=0, dtype={"cas": str})
    
    # Create binary indicators for functional groups
    df['1A1'] = df.iloc[:, :4].apply(lambda row: 1 if 1 in row.values else 0, axis=1)
    df['1A2'] = df.iloc[:, 4:14].apply(lambda row: 1 if 1 in row.values else 0, axis=1)
    df['1A3'] = df.iloc[:, 14:28].apply(lambda row: 1 if 1 in row.values else 0, axis=1)
    df['2B4'] = df.iloc[:, 28:36].apply(lambda row: 1 if 1 in row.values else 0, axis=1)
    
    # Extract and save the target columns
    new_df = df.iloc[:, -5:]
    new_df.to_csv('./data/target_pipeline.csv')
    print("Target dataset prepared and saved.")

def load_dataset(target_csv_name):
    """
    Loads and merges mass spectrometry data with target labels.
    
    Args:
        target_csv_name (str): Path to the target CSV file.
        
    Returns:
        tuple: A tuple containing:
            - total_df (pd.DataFrame): Merged dataset of mass spectra and targets.
            - fn_groups (int): Number of target functional groups.
    """
    # Load mass spectrometry data
    mass_df = pd.read_csv('./data/mass_formula-otherNA-.csv', 
                         index_col=0, dtype={"cas": str})
    
    # Standardize indices to 10-digit zero-padded strings
    mass_df.index = mass_df.index.astype(str).str.zfill(10)
    
    # Load target data
    target_df = pd.read_csv(target_csv_name, index_col=0, dtype={"cas": str})
    target_df.index = target_df.index.astype(str).str.zfill(10)
    
    # Merge datasets on index
    fn_groups = target_df.shape[1]
    total_df = pd.merge(mass_df, target_df, left_index=True, 
                       right_index=True, how='inner')
    
    print(f"Merged dataset shape: {total_df.shape}")
    print(f"Number of functional groups: {fn_groups}")
    
    return total_df, fn_groups

def data_resample(fn_groups, total_df):
    """
    Resamples data to balance class distribution using bootstrap resampling.
    
    Args:
        fn_groups (int): Number of functional groups.
        total_df (pd.DataFrame): Original dataset with imbalanced classes.
        
    Returns:
        pd.DataFrame: Balanced dataset with equal samples per class.
    """
    balanced_data = pd.DataFrame(columns=total_df.columns)
    y = total_df.iloc[:, -fn_groups:]
    
    print(f"Original data shape: {total_df.shape}")
    print("Original class distribution:")
    for label in y.columns:
        print(f"Class {label}: {total_df[label].sum()} samples")
    
    # Determine the maximum class count
    max_count = max([total_df[total_df[col] == 1].shape[0] for col in y.columns])
    
    # Resample each class to match the maximum count
    for label in y.columns:
        label_data = total_df[total_df[label] == 1]
        label_count = label_data.shape[0]
        
        if label_count < max_count:
            # Oversample minority classes
            sampled_data = resample(label_data, replace=True, 
                                  n_samples=max_count, random_state=42)
        else:
            sampled_data = label_data
            
        balanced_data = pd.concat([balanced_data, sampled_data])
    
    print("\nBalanced class distribution:")
    for label in y.columns:
        print(f"Class {label}: {balanced_data[label].sum()} samples")
    
    return balanced_data

def topK(y_val, y_val_pred, k=3):
    """
    Processes predictions using top-K accuracy method.
    
    Args:
        y_val (np.array): True labels.
        y_val_pred (np.array): Predicted probabilities.
        k (int): Number of top predictions to consider.
        
    Returns:
        list: Processed predictions matching true labels where possible.
    """
    y_pred_processed_result_list = []
    
    for i in range(len(y_val)):
        # Get indices of top K predictions
        sorted_indices = sorted(enumerate(y_val_pred[i, :]), 
                              key=lambda x: x[1], reverse=True)[:k]
        
        # Create binary vectors for each of the top K predictions
        top_k_predictions = []
        for idx, _ in sorted_indices:
            pred_vector = np.zeros_like(y_val[i, :])
            pred_vector[idx] = 1
            top_k_predictions.append(pred_vector)
        
        # Default to the top prediction
        y_pred_processed_result = top_k_predictions[0]
        
        # Check if any top K prediction matches the true label
        for pred in top_k_predictions:
            if np.array_equal(y_val[i, :], pred):
                y_pred_processed_result = y_val[i, :]
                break
                
        y_pred_processed_result_list.append(y_pred_processed_result)
    
    return y_pred_processed_result_list

def cal_mean_std(metric_name, value_list):
    """
    Calculates and prints mean, standard deviation, and 95% CI for a list of values.
    
    Args:
        metric_name (str): Name of the metric being calculated.
        value_list (list): List of numerical values to analyze.
    """
    arr = np.array(value_list)
    mean = np.mean(arr)
    std = np.std(arr)
    n = len(arr)
    
    # Calculate 95% confidence interval
    ci_low = mean - 1.96 * (std / np.sqrt(n))
    ci_high = mean + 1.96 * (std / np.sqrt(n))
    
    print(f"{metric_name} mean: {mean:.5f}")
    print(f"{metric_name} 95% CI: [{ci_low:.5f}, {ci_high:.5f}]")