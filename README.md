# Text Classification using Deep Learning

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion) | [Contact](#contact)

## Description
> This is related to Assignment 6: Text Classification using Deep Learning of the course Language Analytics. 

This project aimed to investigate whether lines of Games of Thrones can be classified by the season they belong to. For this classification task two models were trained and evaluated: (1) a baseline model using count-vectorisation and a logistic regression classifier and (2) a deep learning model using pre-trained GloVe word embeddings and a convolutional neural network classifier. While features extracted using count-vectorisation can inform about the words that occur in a given text, they cannot capture anything about the actual meaning of those words and the context that they appear in. Word embeddings are dense feature representations of words, which are learned based on which other words occur around a given word. For instance, if the words house and apartment always occur around words such as live and leave, these words will have similar representations. Consequently, words which are semantically similar will have similar representations. Rather than training these representations from scratch, pre-trained word embeddings, which have been learned using a large corpus of text can be used. These word embeddings can be fed into convolutional neural networks (CNN), which are able to take into account local features, i.e. smaller contexts of text.  

## Methods 

### Data and Preprocessing 
For this project, data from [Kaggle](https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons), containing lines of Games of Thrones and the season they belong to. As the lines varied in their length, the following steps were taken with the aim of cleaning the data and creating more balanced text documents: 
1. Rows in which the text was not a true line, but contained the word SEASON,  EPISODE or CREDITS were removed. 
2. The lines in the data frame still consisted of multiple sentences. Thus, all lines were split into single sentences, while keeping the information about the season. 
4. The single sentences were chunked into chunks of 10 sentences and stored together with their season in a data frame with columns for season and text.

### Count Vectorisazion and Logistic Regression
For the baseline model, the preprocessed data was first split into training and test data using an 75/25 split. Subsequently, features were extracted from the data using count vectorisation. This means, words in the texts are stored in a dictionary, and each text is represented by how often each word of the dictionary occurs in the text. These feature representations were fed into a logistic regression classifier, which was run using default parameters. The model was evaluated based on predictions on the test data. 

### Image Embeddings and Convolutional Neural Networks
For the deep learning model, the preprocessed data was also split into training and test data using an 75/25 split. First, the labels were binarised to be fed into the CNN. Afterwards, the texts were tokenised, meaning they were turned into a vectors. To be fed into the CNN, it is required that all vectors have the same length. Thus, the vectors were padded to be as long as the maximum vector. These vectors were then used to create an embedding matrix, based on the pre-trained GloVe embeddings of dimension 100. This embedding matrix the input of the first layer, the embedding layer of the CNN. Even though the pre-trained GloVe embedding were used, they were allowed to be re-trained  based on the input data (set to trainable). Following this embedding layer was a convolutional layer (128 nodes, with relu-activation function, L2 regularisation of 0.001), a global max-pooling layer, a drop out layer (0.02), a fully connected layer (32 nodes, with relu-activation function, L2 regularisation of 0.001), another drop-out layer (0.02) and finally a layer to classify the line to be one of the eight seasons. The drop-out layers and regularisation method were used to reduce overfitting of the model on the training data. For more details see the [model visualisation](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/2_cnn_classifier/pretrained_100_e20/cnn_model.png) and [model summary](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/2_cnn_classifier/pretrained_100_e20/cnn_summary.txt). The model was trained for 20 epochs using a batch size of 20.

## Repository Structure 
```
|-- data/
    |-- glove/                          # Empty folder, for CNN to be filled with GloVe files, see instructions below
    |-- Game_of_Thrones_Script.csv      # Raw data of GoT transcripts
    
|-- out/                                # Directory for output, corresponding to scripts
    |-- 0_preprocessing/                # Directory for output of script 0_preprocessing.py
        |-- GoT_preprocessed_10.csv     # Preprocessed data, with chunks of 10 sentences
    |-- 1_lr_classifier/                # Directory for output of script 1_lr_classifier.py
        |-- lr_metrics.txt              # Classification report of logistic regression classifier
        |-- lr_matrix.png               # Classification matrix of logistic regression classifier
    |-- 2_cnn_classifier/               # Directory for output of script 2_cnn_classifier.py
        |-- cnn_summary.txt             # Summary of CNN architecture
        |-- cnn_model.png               # Visualisation of CNN architecture
        |-- cnn_history.png             # Visualisation of CNN training history
        |-- cnn_metrics.txt             # Classification report of CNN
        |-- cnn_matrix.png              # Classification matrix of CNN

|-- src/                                # Directory containing main scripts of the project
    |-- 0_preprocessing.py              # Script for preprocessing raw data
    |-- 1_lr_classifier.py              # Script for logistic regression classifier
    |-- 2_cnn_classifier.py             # Script for CNN classifier with word embeddings
   
    
|-- README.md
|-- create_venv.sh                       # Bash script to create virtual environment
|-- requirements.txt                     # Dependencies, installed in virtual environment

```

## Usage 
**!** The scripts have only been tested on Linux, using Python 3.6.9. 

### 1. Cloning the Repository and Installing Dependencies
To run the scripts in this repository, I recommend cloning this repository and installing necessary dependencies in a virtual environment. The bash script `create_venv.sh` can be used to create a virtual environment called `venv_cnn` with all necessary dependencies, listed in the `requirements.txt` file. The following commands can be used:

```bash
# cloning the repository
git clone https://github.com/nicole-dwenger/cdslanguage-cnn.git

# move into directory
cd cdslanguage-cnn/

# install virtual environment
bash create_venv.sh

# activate virtual environment 
source venv_cnn/bin/activate
```

### 2. Data and Pretrained Embeddings
The Games of Thrones Data, which was downloaded from [Kaggle](https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons) is already in the repository, in the `data/` directory. However, to run the CNN, it is necessary to download the pretrained GloVe word embeddings. This should be done in the `data/glove/` directory (which is already prepared, but empty). After following the steps below, you should have four .txt files in the glove directory: 

```bash
# move into glove directory
cd data/glove/

# download pretrained glove embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip

# unzip files
unzip unzip -q glove.6B.zip

# move back into root directory
cd ../../
```

### 3. Scripts
This repository contains three scripts: `0_preprocessing.py`, `1_lr_classifier.py` and `2_cnn_classifier.py`. Note that to run the scripts `1_lr_classifier.py` and `2_cnn_classifier.py` it is necessary to first preprocess the raw data using `0_preprocessing.py`. Detailed descriptions of how to run each of the scripts are outlined below. 

### 3.0. Preprocessing of GoT Data: 0_preprocessing.py
The script `0_preprocessing.py` preproceses the GoT data, following the steps outlined above. The script should be called from the `src/` directory:

```bash
# move into src
cd src/

# run script with default parameters
python3 0_preprocessing.py

# run script with specified parameters
python3 0_preprocessing.py -c 20
```
__Parameters:__

- `-c, --chunk_size`: *int, optional, default:* `10`\
   Number of sentences to be in one chunk. 
    
__Output__ saved in `out/0_preprocessing/`: 

- `GoT_preprocessed_{chunk_size}`\
   Preprocessed GoT data as .csv file with column "season" and "text". 
   
### 3.1. Logistic Regression Classifier: 1_lr_classifier.py
The script `1_lr_classifier.py` trains and evaluates a logistic regression classifier using extract count vectors. The script should be called from the `src/` directory:

```bash
# move into src
cd src/

# run script with default parameters
python3 1_lr_classifier.py

# run script with specified parameters
python3 1_lr_classifier.p -i ../out/0_preprocessing/GoT_preprocssed_20.csv
```
__Parameters:__

- `-i, --input_file`: *str, optional, default:* `../out/0_preprocessing/GoT_preprocessed_10.csv`\
   Input file, should be data, which was preprocessed with `0_preprocessing.py`, with columns of `season` and `text`. 
    
__Output__ saved in `out/1_lr_classifier/`: 

- `lr_metrics.txt`\
   Classification report of logistic regression classifier. Filename is enumerated if it exists already.

- `lr_matrix.png`\
   Classification/confusion matrix of logistic regression classifier. Filename is enumerated if it exists already.
   
   
### 3.2. CNN Classifier: 2_cnn_classifier.py
The script `2_cnn_classifier.py` trains and evaluates a CNN using pre-trained GloVe word embeddings. The script should be called from the `src/` directory:

```bash
# move into src
cd src/

# run script with default parameters
python3 2_cnn_classifier.py

# run script with specified parameters
python3 2_cnn_classifier.p -e 10
```
__Parameters:__

- `-i, --input_file`: *str, optional, default:* `../out/0_preprocessing/GoT_preprocessed_10.csv`\
   Input file, should be data, which was preprocessed with `0_preprocessing.py`, with columns of `season` and `text`. 
   
- `-e, --epochs:` *int, default:* `20`\
  Number of epochs to train the model.
  
- `-b, --batch_size`: *int, default:* `20`\
  Size of batches to train model on.
  
- `-ed, --embedding_dim`: *int, default:* `100`\
  Embedding dimension, either `50`, `100`, `200` or `300`. 
    
__Output__ saved in `out/2_cnn_classifier/`: 

- `cnn_summary.txt`\
   Model summary, i.e. information about layer architecture. Filename is enumerated if it exists already.
   
- `cnn_model.png`\
   Visualisation of model architecture. Filename is enumerated if it exists already.
   
- `cnn_history.png`\
   Visualisation of model training history. Filename is enumerated if it exists already.

- `cnn_metrics.txt`\
   Classification report of CNN. Filename is enumerated if it exists already.

- `cnn_matrix.png`\
   Classification/confusion matrix of CNN. Filename is enumerated if it exists already.


## Results and Discussion 
All results can be found in the `out/` directory of this repository. The [classification report](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/1_lr_classifier/lr_metrics.txt) of the logistic regression classifier indicated, that the model achieved a weighted F1 score of 0.35. The [classification report](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/2_cnn_classifier/cnn_metrics.txt) of the CNN indicated a weighted F1 score of 0.31. Thus, only considering the F1 score, the CNN could not really outperform the base line, logistic regression classifier. 

The confusion matricies can be provide more detailed information into which seasons where mixed up or more difficult to classifiy. Both the logistic regression classifier and the CNN performed best at classifying season 1 and season 7. However, there may be other reasons, e.g. disbalanced data for these results. 

Logistic Regression             | CNN
:-------------------------:|:-------------------------:
![](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/1_lr_classifier/lr_matrix.png)  |  ![](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/2_cnn_classifier/cnn_matrix.png)

Looking closer at the training history of CNN, it can be seen that despite regularisation and drop-out layers the model started overfitting at around epoch 10. This is indicated by the fact that the validation loss is diverging, meaning it increases, while the training loss continues to decrease. 

<p align="center">
  <img width="350" src="https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/2_cnn_classifier/cnn_history.png">
</p>

Overall, it seems like this data may have been too complex to classify using word embeddings. Other approaches, e.g. using NER may be more useful to classify seasons, as they may be a simpler indicator of which season lines belong to. 

## Contact
If you have any questions, feel free to contact me at 201805351@post.au.dk. 