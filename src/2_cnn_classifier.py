#!/usr/bin/python

"""
CNN classification of preprocessed GoT data, which has sentences chunked into 10 sentences, 
using pretrained GloVe embeddings, which can be used as pretrained, or be re-trained on the data

Optional input:
  - -i, --input_file: str, default: "../out/0_preprocessing/GoT_preprocessed_10.csv", input csv with "season" and "text"
  - -e, --epochs: int, default: 20, number of epochs to train the model
  - -b, --batch_size: int, default: 20, size of batches to train model on 
  - -ed, --embedding_dim: int, default: 100, embedding dimension, either 50,100,200,300

Output saved in out/2_cnn_classifier/:
- cnn_summary.txt: model summary of cnn model
- cnn_model.png: visualisation of the model architecture
- cnn_history.png: history plot of the cnn model
- cnn_report.txt: classification report of the model
- cnn_matrix.png: png of classification / confusion matrix
"""

# LIBRARIES --------------------------------------------

# Basics
import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Utils
import sys
sys.path.append(os.path.join(".."))
from utils.clf_utils import (get_train_test, binarize_labels, create_embedding_matrix,
                             tokenize_texts, pad_texts, int_to_labels, classification_matrix,
                             save_model_info, save_model_history, save_model_report, save_model_matrix)


# ML tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# CNN, tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Dense, Embedding, Flatten, GlobalMaxPool1D, 
                                     Conv1D, Dropout, MaxPool1D)


# MAIN FUNCTION ----------------------------------------

def main():
    
    # ---ARGUMENT PARSER ---
    
    # Initialise argument parser
    ap = argparse.ArgumentParser()
    
    # Input option for number of epochs
    ap.add_argument("-i", "--input_file", type=str, help="Path to input data file, csv with columns 'season' and 'text'", 
                    required=False, default="../out/0_preprocessing/GoT_preprocessed_10.csv")
    
    # Input option for number of epochs
    ap.add_argument("-e", "--epochs", type=int, help="Number of epochs",
                    required=False, default=20)
    
    # Input option for batch size
    ap.add_argument("-b", "--batch_size", type=int, help="Batch size",
                    required=False, default=20)
    
    # Input option for embedding dimensions
    ap.add_argument("-ed", "--embedding_dim", type=int, help="Size of embedding dimension",
                     required=False, default=100)

    # Retrieve input arguments
    args = vars(ap.parse_args())
    input_file = args["input_file"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    embedding_dim = args["embedding_dim"]
    
    # --- PREPARE DATA ---
    
    print(f"\n[INFO] Initialising CNN classifier for {input_file}")
    
    # Read input data to df
    df = pd.read_csv(input_file)
    
    # Get train and test values from df
    X_train, X_test, y_train, y_test = get_train_test(df, "text", "season", test_size=0.25)

    # Binarizse labels
    y_train_binary, y_test_binary, label_names = binarize_labels(y_train, y_test)
    
    # Tokenize texts, and keeping the defined number of words
    X_train_toks, X_test_toks, vocab_size, word_index = tokenize_texts(X_train, X_test, num_words=4000)
    
    # Pad texts, for all to have the same length
    X_train_pad, X_test_pad, max_len = pad_texts(X_train_toks, X_test_toks)
    
    # --- EMBEDDING MATRIX ---
    
    # Define embedding matrix using pre-trained GlovEmbeddings
    embedding_matrix = create_embedding_matrix(f'../data/glove/glove.6B.{embedding_dim}d.txt', 
                                               word_index, embedding_dim)

    # --- CNN ---
    
    # Define CNN
    model = Sequential()
    
    # Add embedding layer
    model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], 
                        input_length=max_len, trainable=True))
    
    # Add convolutional layer
    model.add(Conv1D(128, 5, activation='relu', kernel_regularizer=L2(0.001)))
    # Add max-pooling layer
    model.add(GlobalMaxPool1D())
    # Add drop out layer
    model.add(Dropout(0.2))
    # Add fully conencted layer
    model.add(Dense(32, activation='relu', kernel_regularizer=L2(0.001)))
    # Add drop out layer
    model.add(Dropout(0.2))
    # Add output classification layer
    model.add(Dense(8, activation='softmax'))

    # Compile CNN
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train CNN
    history = model.fit(X_train_pad, y_train_binary,
                        validation_data = (X_test_pad, y_test_binary),
                        epochs=epochs, batch_size=batch_size,
                        verbose=1)
    
    # Get classification report
    predictions = model.predict(X_test_pad, batch_size)
    report = classification_report(y_test_binary.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names = label_names)
                              
    # Turn binarised labels into true labels
    labels_true, labels_pred = int_to_labels(y_test_binary, predictions, label_names)  
    # Get classifiction matrix
    matrix = classification_matrix(labels_true, labels_pred)
    
    # --- OUTPUT ---
    
    # Print classification report
    print("[OUTPUT] Classification report for CNN classifier:\n")
    print(report)
    
    # Prepare output directory
    out_directory = os.path.join("..", "out", "2_cnn_classifier")
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    
    # Save outputs
    save_model_info(model, out_directory, "cnn_summary.txt", "cnn_model.png")
    save_model_history(history, epochs, out_directory, "cnn_history.png")
    save_model_report(report, input_file, out_directory, f"cnn_metrics.txt")
    save_model_matrix(matrix, out_directory, f"cnn_matrix.png")

          
if __name__=="__main__":
    main()
    
  