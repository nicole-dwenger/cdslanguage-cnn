#!/usr/bin/env python

"""
Script to preprocess the GoT data: 
  - Clear non-sentences,
  - Split all texts into single sentences and chunk them into chunks of 10

Input:
  - -c, chunk_size: int, optional, default: 10, number of sentences to be in chunk

Output, saved in out/:
  - 0_preprocessed_GoT_{chunk_size}.csv: csv file with column names "season" and "text", 
    where text contains chunks of sentences
  
"""

# LIBRARIES ------------------------------------

# Basics
import os
import argparse
from tqdm import tqdm

# Data
import pandas as pd
import numpy as np

# NLP
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])


# HELPER FUNCTIONS -----------------------------

def split_sentences(text):
    """
    For a text, split text into individual sentences. 
    Input:
      - text: str of text
    Returns:
      - sentences: list of sentences of text
    """
    # Apply spacy language model
    doc = nlp(text)
    # Split sentences into list
    sentences = [sent.string.strip() for sent in doc.sents]

    return sentences

def chunk_sentences(sentences, chunk_size):
    """
    For a list of sentences, chunk sentences.
    Input:
      - sentences: list of single sentences
      - chunk_size: n sentences to be in one chunk
    Returns:
      - list of chunks, each containing n sentences
    """
    # Create empty target list for chunks
    chunks = []
    # Chunks into chunk size, and append to chunks list
    for i in range(0, len(sentences), chunk_size):
        chunks.append(' '.join(sentences[i:i+chunk_size]))  
        
    return chunks


# MAIN FUNCTION --------------------------------

def main():
    
    # --- ARGUMENT PARSER ---
    
    # Argument option for chunk size
    ap = argparse.ArgumentParser()
    # input option for filepath to the input file, with default of realnews data
    ap.add_argument("-c", "--chunk_size", required=False, help="Size of chunks of sentences", default=10, type=int)

    # Get the input
    args = vars(ap.parse_args())
    chunk_size = args["chunk_size"]
    
    # --- DATA PROCESSING ---
    
    # Print start message
    print(f"\n[INFO] Initialising preprocessing of GoT data, cleaning and chunking {chunk_size} sentences.")
     
    # Reading GoT data
    file = os.path.join("..", "data", "Game_of_Thrones_Script.csv")
    data = pd.read_csv(file)
    
    # Removing rows in which sentence is one of these not_sentences
    not_sentence = ["COLD OPEN", "CREDITS", "EPISODE", "SEASON"]
    cleaned = data[data["Sentence"].isin(not_sentence) == False]
    
    # Create empty target dataframe for processed data  
    out_df = pd.DataFrame(columns = ["season", "text"])

    # Within each season: split all sentences, create chunks of 10 sentences
    # For each unique season
    for season in tqdm(cleaned["Season"].unique()):
        
        # Get a dataframe of only rows belonging to the season
        season_df = cleaned[cleaned["Season"] == season]
        # Put all texts into a list
        season_texts = season_df["Sentence"].tolist()

        # Create empty list of all single sentences
        sentences = []
        # For each of the texts in the list of texts of the season
        for text in season_texts:
            # Split the texts into single sentences
            split = split_sentences(text)
            # Append to senteces list
            sentences.extend(split)

        # Chunk the single sentences into chunks of chunk size
        chunks = chunk_sentences(sentences, chunk_size)
        # For each chunk, write the season and the text into a row in the df
        for chunk in chunks:
            out_df = out_df.append({"season": season, 
                                    "text": chunk}, ignore_index = True)
            
    # --- OUTPUT ---
    
    # Create output directory
    out_directory = os.path.join("..", "out", "0_preprocessing")
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
       
    # Save output dataframe
    out_path = os.path.join(out_directory, f"GoT_preprocessed_{chunk_size}.csv")
    out_df.to_csv(out_path)
    
    # Print done message
    print(f"Done! Preprocessed data saved in {out_path}")
                                  
   
if __name__=="__main__":
    main()