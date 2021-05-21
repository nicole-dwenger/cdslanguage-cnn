#!/usr/bin/python

"""
Logistic regression classification of preprocessed GoT data, with columns "season" and "text"

Steps:
  - Read preprocessed data
  - Get X (chunked sentences) and y (season labels)
  - Split into train and test data using 80/20 split
  - Use count vectoriser to extract features from X
  - Train and evaluate logistic regression classifier
  - Get and save classification report and matrix 
  
Input:
  - -i, --input_file: str, optional, default: ../out/0_preprocessing/GoT_preprocessed_10.csv

Output saved in ../out/1_lr_classifier/
  - lr_metrics.csv: classification report
  - lr_matrix.png: classification matrix
"""

# LIBRARIES -------------------------------------------

# Basics
import os
import argparse

# Utils
import sys
sys.path.append(os.path.join(".."))
from utils.clf_utils import classification_matrix, save_model_report, save_model_matrix

# Data and Visualisation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn for ML
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# MAIN FUNCTION ---------------------------------------

def main():
    
    # --- ARGUMENT PARSER ---
    
    # Initialise argument aprser
    ap = argparse.ArgumentParser()
    
    # Input option for input file 
    ap.add_argument("-i", "--input_file", type=str, help="Path to input data file, csv with columns 'season' and 'text'", 
                    required=False, default="../out/0_preprocessing/GoT_preprocessed_10.csv")

    # Retrieve inputs
    args = vars(ap.parse_args())
    input_file = args["input_file"]
    
    # Prepare output directory
    out_directory = os.path.join("..", "out", "1_lr_classifier")
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    
    # --- PREPARE DATA ---
    
    print(f"\n[INFO] Initialising logistic classifier for {input_file}")
    
    # Read input data to df
    df = pd.read_csv(input_file)
    # Save texts (X) and labels (y) in lists
    X = df["text"].values
    y = df["season"].values
    
    # Split into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Extract count features 
    vectorizer = CountVectorizer()
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    # --- LOGISTIC REGRESSION CLASSIFIER ---
    
    print("[INFO] Training and evaluating logistic regression classifier.")
    
    # Train classifier
    clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_features, y_train)
    
    # Evaluate classifier: generate predictions
    predictions = clf.predict(X_test_features)
    
    # Get and save classification report
    report = classification_report(y_test, predictions)
    save_model_report(report, input_file, out_directory, f"lr_metrics.txt")
    
    # Get and save matrix
    matrix = classification_matrix(y_test, predictions)
    save_model_matrix(matrix, out_directory, f"lr_matrix.png")
    
    # Print classification report
    print("[OUTPUT] Classification report for logistic regression classifier:\n")
    print(report)
    
    # Print done message
    print(f"\n[INFO] All done, output saved in {out_directory}!")

        
if __name__=="__main__":
    main()
    