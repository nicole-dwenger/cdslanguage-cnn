#!/usr/bin/python

"""
Utility script with functions used in lr classifier and cnn classifier. 

For data preparation: 
  - get_train_test(): from dataframe, and specified columns, get train and test data and labels
  - tokenize_text(): tokenize a list of texts, and return tokenized texts
  - pad_texts(): add padding to texts, for them to have equal lengths

For CNN model output:
  - int_to_labels(): turn enumerated labels into corresponding text labels, e.g. 0 = season1, 
  - classification_matrix(): plot classification matrix, i.e. confusion matrix
  - unique_path(): enumerates filename, if file exists already
  - save_model_info(): save information of model layers and visualisation of model
  - save_model_history(): save plot of model training history
  - save_model_report(): save txt file of classification report
  - save_model_matrix(): save classification matrix
"""

# LIBRARIES ------------

# Basics
import os
import numpy as np
import pandas as pd
from contextlib import redirect_stdout

# Sklearn ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Tensorflow CNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model

# Visualisations
import matplotlib.pyplot as plt
import seaborn as sns


# DATA PREPARATION --------------------------------------

def get_train_test(df, text_column, label_column, test_size):
    """
    From a dataframe, extract texts and labels, and split into test and train
    Input:
      - df: dataframe with texts and labels
      - text_column: name of column in df storing text documents (X)
      - label_column: name of column in df storing labels (y)
      - test_size: size of test split
    Returns:
      - train and text X (texts) and y (labels)
    """
    # Extract texts (X) and labels (y) from columns of df
    X = df[text_column].values
    y = df[label_column].values
    
    # Split test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42) 
    
    return X_train, X_test, y_train, y_test

def binarize_labels(y_train, y_test):
    """
    Binarize test and train labels
    Input:
      - y_train, y_test: training and test labels as names
    Returns:
      - y_train_binary, y_test_binary: binarized labels
      - label_names: list of unique, sorted label names
    """
    # Initialise binariser
    lb = LabelBinarizer()
    
    # Apply binarizer to train and test data
    y_train_binary = lb.fit_transform(y_train)
    y_test_binary = lb.fit_transform(y_test)
    
    # Get the sorted, unique label names
    label_names = sorted(set(y_train))
    
    return y_train_binary, y_test_binary, label_names
  
def tokenize_texts(X_train, X_test, num_words):
    """
    Tokenizing the documents/texts, and getting the vocabulary size 
    Input: 
      - X_train: array of texts used for training 
      - X_test: array of texts used for testing
      - num_words, refers to the maximum number of most common words to keep 
    Returns: 
      - X_train_toks: array of tokenized training texts
      - X_test_toks: array of tokenized test texts
      - vocab_size: size of vocabulary
    """
    # Intialise tokenizer
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X_train)

    # Tokenized training and test data
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)

    # Overall vocabulary size
    vocab_size = len(tokenizer.word_index) + 1 
    
    # Get word indicise
    word_index = tokenizer.word_index
    
    return X_train_toks, X_test_toks, vocab_size, word_index

def pad_texts(X_train_toks, X_test_toks):
    """
    Add padding to tokenized texts, to ensure that they have the same length
    Get maximum length of all documents, append 0s to documents for max length 
    Input: 
      - X_train_toks: array of tokenized training texts
      - X_test_toks: array of tokenized test texts
    Returns:
      - X_train_pd: array of padded, tokenized training texts
      - X_test_pd: array of padded, tokenized test texts
      - max_len: length of longest text, now length of all texts
    """
    # Get the maximum length of the test and train tokens separately
    max_train = len(max(X_train_toks, key=len))
    max_test = len(max(X_test_toks, key=len))
    max_len = max(max_train, max_test)

    # Apply padding to training and test tokens
    X_train_pad = pad_sequences(X_train_toks, padding='post', maxlen=max_len)
    X_test_pad = pad_sequences(X_test_toks, padding='post', maxlen=max_len)
    
    return X_train_pad, X_test_pad, max_len

# CNN EMBEDDINGS --------------------------------------

def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    Read GloVe embeddings and generate an embedding matrix
    Input:
      - filepath: path to glove embeddings file
      - word_index: indicies, extracted from tokenizer
      - embedding_dim: dimension of keras impedding, and glove embedding
    Output:
      - matrix of embeddings for each word
    """
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


# CNN MODEL OUTPUTS --------------------------------------

def int_to_labels(actual_binary, predictions_binary, label_names):
    """
    Transforms binarised/enumerated lables back into their original names
    Input:
      - actual_binary: list of true, binarised labels
      - predictions_binary: list of predictions for labels
      - label_names: list of sorted, unqiue names
    Returns:
      - labels_ture, labels_pred: corresponding original label names
    """
    # Get the int (number) of the label
    int_actual = (actual_binary.argmax(axis=1)).tolist()
    int_pred = (predictions_binary.argmax(axis=1)).tolist()

    # For each of the int of labels, get the corresponding name
    # For true labels
    labels_actual = []
    for i in range(0, len(int_actual)):
        labels_actual.append(label_names[int_actual[i]])
    # For predicted labels
    labels_pred = []
    for i in range(0, len(int_pred)):
        labels_pred.append(label_names[int_pred[i]])
        
    # Turn lists into arrays
    labels_actual = np.array(labels_actual)
    labels_pred = np.array(labels_pred)
    
    return labels_actual, labels_pred

def classification_matrix(actual, predictions):
    """
    Function to plot classification matrix
    Input:
      - actual: array of actual label names
      - predictions: array of predicted label names
    Returns:
      - classification_matrix
    """
    # Create confusion matrix
    cm = pd.crosstab(actual, predictions, rownames=['Actual'], 
                     colnames=['Predicted'], normalize='index')
    
    # Initialise figure
    p = plt.figure(figsize=(10,10));
    # Plot confusion matrix on figure as heatmap
    p = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)
    
    # Save the figure in variable
    classification_matrix = p.get_figure()
        
    return classification_matrix

def unique_path(filepath):
    """
    Create unique filename by enumerating if path exists already 
    Input:
      - filepath: desired fielpath
    Returns:
      - new_path: enumerated if it exists already
    """ 
    # If the path does not exist
    if not os.path.exists(filepath):
        # Keep the original filepath
        return filepath
    
    # If path exists:
    else:
        i = 1
        # Split the path and append a number
        path, ext = os.path.splitext(filepath)
        # Add extension
        new_path = "{}_{}{}".format(path, i, ext)
        
        # If the extension exists, enumerate one more
        while os.path.exists(new_path):
            i += 1
            new_path = "{}_{}{}".format(path, i, ext)
            
        return new_path
    
def save_model_info(model, output_directory, filename_summary, filename_plot):
    """
    Save model summary in .txt file and plot of model in .png
    Input:
      - model: compiled model
      - output_directory: path to output directory
      - filename_summary: name of file to save summary in
      - filename_plot: name of file to save visualisation of model
    """
    # Define path fand filename for model summary
    out_summary = unique_path(os.path.join(output_directory, filename_summary))
    # Save model summary in defined file
    with open(out_summary, "w") as file:
        with redirect_stdout(file):
            model.summary()

    # Define path and filename for model plot
    out_plot = unique_path(os.path.join(output_directory, filename_plot))
    # Save model plot in defined file
    plot_model(model, to_file = out_plot, show_shapes = True, show_layer_names = True)

def save_model_history(history, epochs, output_directory, filename):
    """
    Plotting the model history, i.e. loss/accuracy of the model during training
    Input: 
      - history: model history
      - epochs: number of epochs the model was trained on 
      - output_directory: desired output directory
      - filename: name of file to save history in
    """
    # Define output path
    out_history = unique_path(os.path.join(output_directory, filename))

    # Visualize history
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_history)
    
def save_model_report(report, input_file, output_directory, filename):
    """
    Save report to output directory
    Input: 
      - report: model classifcation report
      - output_directory: final output_directory
      - filename: name of file to save report in
    """
    # Define output path and file for report
    report_out = unique_path(os.path.join(output_directory, filename))
    # Save report in defined path
    with open(report_out, 'w', encoding='utf-8') as file:
        file.writelines(f"Classification report for model trained on {input_file}:\n")
        file.writelines(report) 

def save_model_matrix(matrix, output_directory, filename):
    """
    Save model matrix in outptut directory
    Input:
      - matrix: plot of classification matrix
      - output_directory: path to output directory
      - filename: desired filename
    """
    out_matrix = unique_path(os.path.join(output_directory, filename))
    matrix.savefig(out_matrix)
    
if __name__=="__main__":
    pass  