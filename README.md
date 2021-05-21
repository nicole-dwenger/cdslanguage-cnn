# Assignment 5: Unsupervised Machine Learning

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion) | [Contact](#contact)


## Description
> This is related to Assignment 6: Text Classification using Deep Learning of the course Language Analytics. 

The meaning of text is not only defined by occurance of individual words (which could be captured through e.g. count-vectorization or tfidf-vectorization), and sparse one-hot encoded vectors, but also through the actual meaning of words and the context that those individual words occur in. 

**Word embeddings** are dense feature representations of words, where semantically similar words have similar representations. These word embeddings are learned from a corpus of text documents, based on around which other words they occur. Thus, if two words occur always together with other words, they will have similar representations. Rather than training these representations from scratch, pre-trained word embeddings can be used, which have been extracted from a large corpus of text. In this project, GloVe embeddings are used, which were trained using Wikipedia 2014 + Gigaword 5 (more info [here](https://nlp.stanford.edu/projects/glove/)). 

**Convolutional Neural Networks** have the advantage over e.g. traditional machine learning models, that they can not only take into account global features across a text, but also local features, i.e. smaller contexts. As a first layer, CNNs can take an embedding layer. If no pre-trained embeddings are used, this layer will learn to word embeddings using the input data. However, if pre-trained embeddings are used, the words of the text can be transformed into these pre-trained embeddings and fed into the model. Additionally, one can also update or re-train the pretrained embeddings based on the text inut. 

This project aimed to investigate whether lines of Games of Thrones can be classified by the season they belong to. Thus, this repository contains two scripts, (1) to train and evaluate a base line model, using count-vectorisation and a logistic regression classifier and (2) to train and evalute a CNN, using pre-trained GloVe embeddings. 


## Methods 

### Data and Preprocessing 
The data, which was used for this project contains lines of Games of Thrones and information about the season they belong to. The data was extracted from [Kaggle](https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons) and preprocessed with the following steps, aiming to create more balanced text document, e.g. of similar length:
1. Remove rows in which the `Sentence` was not a sentence, but `SEASON` or `EPISODE` or `CREDITS` 
2. The `Sentence` column still contained multiple sentences, i.e. when one of the characters was talking for a longer time. All sentences were split into single sentences. 
3. The single sentences were chuncked in chunks of 10 sentences and stored together with the season information in a dataframe with columns `season` and `text`

### Count Vectorisazion and Logistic Regression
As a baseline model, features were extracted from each of the texts using count vectorisation. This means, that that words in the texts are stored in a dictionary, and each text is then represented as how often each word of the dictionary occurs in the text. The logistic regression classifier used these extracted feature spaces to classify the text. For this project it was run with L2 regularisation, and a max of 1000 iterations. 

### Image Embeddings and Convolutional Neural Networks
The CNN had the following layer structure: 

- Embedding Layer: embedding matrix of GloVe embeddings, using embedding dimension of 100
- Convolutional layer with 128 nodes, relu activation and L2 regularisation of 0.001
- Global max-pooling layer 
- Drop out ayer of 0.02
- Fully connected layer with 32 nodes, relu activation and L2 regularisation of 0.001
- Drop out layer of 0.02
- Output, classification layer

The drop out layers and regularisation methods were used, to reduce overfitting of the model on the training data. For more details see also the [model visualisation](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/2_cnn_classifier/pretrained_100_e20/cnn_model.png) and [model summary](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/2_cnn_classifier/pretrained_100_e20/cnn_summary.txt). The model was once trained using the pre-trained word embeddings, and once allowing the pre-trained word embeddings to be updated with the input data (i.e. trainble). In both cases, the model was trained for 20 epochs using a batch size of 20. 

## Repository Structure 
```
|-- data/
    |-- glove/                          # Empty folder, for CNN to be filled with GloVe files, see instructions below
    |-- Game_of_Thrones_Script.csv      # Raw data of GoT transcripts
    
|-- out/                                # Directory for output, corresponding to scripts
    |-- 0_preprocessing/                # Directory for output of script 0_preprocessing.py
        |-- GoT_preprocessed_10.csv     # Preprocessed data, with chunks of 10 sentences
    |-- 1_lr_classifier/                # Directory for output of script 1_lr_classifier.py
        |-- lr_metrics.txt              
        |-- lr_matrix.png               
    |-- 2_cnn_classifier/               # Directory for output of script 2_cnn_classifier.py
        |-- pretrained_100_e20/         # Directory for output of CNN with pretrained embeddings of size 100
            |-- cnn_summary.txt
            |-- cnn_model.png
            |-- cnn_history.png
            |-- cnn_metrics.txt
            |-- cnn_matrix.png
        |-- trained_100_e20/            # Directory for output of CNN with re-trained embedddings of 100
            |-- ...

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
The Games of Thrones Data, which was downloaded from [Kaggle](https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons) is already in the repository, in the `data/` directory. However, to run the CNN, it is necessary to download the pretrained GloVe word embeddings. This should be done in the `data/glove/` directory (which is already prepared, but empty). After following the steps below, you should have 4 .txt files in the glove directory: 

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
   Classification report of logistic regression classifier. Filename is enumerated if it exists already, to avoid overwriting.

- `lr_matrix.png`\
   Classification/confusion matrix of logistic regression classifier. Filename is enumerated if it exists already, to avoid overwriting.
   
   
### 3.2. CNN Classifier: 2_cnn_classifier.py
The script `2_cnn_classifier.py` trains and evaluates a CNN using pre-trained GloVe word embeddings. The script should be called from the `src/` directory:

```bash
# move into src
cd src/

# run script with default parameters
python3 2_cnn_classifier.py

# run script with specified parameters
python3 2_cnn_classifier.p -o cnn_1
```
__Parameters:__

- `-o, --output_name`: *str,* **required**\
   Name of directory, in which all outputs of the model will be stored. The directory will be created in `out/` when running the script. 

- `-i, --input_file`: *str, optional, default:* `../out/0_preprocessing/GoT_preprocessed_10.csv`\
   Input file, should be data, which was preprocessed with `0_preprocessing.py`, with columns of `season` and `text`. 
   
- `-e, --epochs:` *int, default:* `20`\
  Number of epochs to train the model.
  
- `-b, --batch_size`: *int, default:* `20`\
  Size of batches to train model on.
  
- `-ed, --embedding_dim`: *int, default:* `100`\
  Embedding dimension, either `50`, `100`, `200` or `300`. 
  
- `-et, --embedding_trainable`: *bool, default:* `False`\
   To make embedding weights trainable, add `--embedding_trainable` (without any further parameters) when running the script. 
    
__Output__ saved in `out/2_cnn_classifier/{output_name}`: 

- `cnn_summary.txt`\
   Model summary, i.e. information about layer architecture. 
   
- `cnn_model.png`\
   Visualisation of model architecture. 
   
- `cnn_history.png`\
   Visualisation of model training history. 

- `cnn_metrics.txt`\
   Classification report of CNN. 

- `cnn_matrix.png`\
   Classification/confusion matrix of CNN. 


## Results and Discussion 
All results can be found in the `out/` directory of this repository. The [classification report](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/1_lr_classifier/lr_metrics.txt) of the logistic regression classifier indicates, that the model achieved a weighted F1 score of 0.33. Classification reports ([pretrained embeddings](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/2_cnn_classifier/pretrained_100_e20/cnn_metrics.txt), [retrained embeddings](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/2_cnn_classifier/trained_100_e20/cnn_metrics.txt)) of the CNNs indicate, that the model using pre-trained embeddings, achieved a weighted F1 score of 0.33, while the model in which embedding weights were re-trained achieved a weighted F1 score of 0.34. Thus, only considering the F1 score, the CNN could not really outperform the base line, logistic regression classifier. 

The model histories of the CNNs indicate, that despite regularisation and drop-out layers the model started overfitting at around epoch 10. This is indicated by the fact that the validation loss is diverging, meaning it increases, while the training loss continues to decrease. 

Pretrained Embeddings             | Retrained Embeddings
:-------------------------:|:-------------------------:
![](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/2_cnn_classifier/pretrained_100_e20/cnn_history.png)  |  ![](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/2_cnn_classifier/trained_100_e20/cnn_history.png)


The confusion matricies can be provide more detailed information into which seasons where mixed up or more difficult to classifiy. Both the logistic regression classifier and the CNNs performed best at classifying season 1, season 2 and season 7. However, there may be other reasons, e.g. disbalanced data for these results. 

Logistic Regression             | Retrained Embeddings
:-------------------------:|:-------------------------:
![](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/1_lr_classifier/lr_matrix.png)  |  ![](https://github.com/nicole-dwenger/cdslanguage-cnn/blob/master/out/2_cnn_classifier/trained_100_e20/cnn_matrix.png)


Overall, it seems like this data may have been too complex to classify using word embeddings. Other approaches, e.g. using NER may be more useful to classify seasons, as they may be a simpler indicator of which season lines belong to. 

