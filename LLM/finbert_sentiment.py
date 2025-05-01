import os
import re
import pandas as pd
import torch
import torch.nn.functional as F
import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set the directory containing the CSV files
dir_path = '/home/r_huang3/GraLNA/New/LLM/annualreport/1993/train'

# Load the FinBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

# Define the sentiment classes and scores
sentiment_classes = ['positive', 'neutral', 'negative']
sentiment_scores = [1, 0, -1]

# Initialize an empty list to store the results
results_doc = []

# Iterate through all the text files in the directory
for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(dir_path, filename)

        # Read the contents of the text file
        with open(file_path, 'r') as file:
            text = file.read().replace('\n', '')

        # Tokenize the text using the BERT tokenizer
        tokens = tokenizer.encode_plus(text, max_length=512, truncation=True, padding='max_length',
                                       add_special_tokens=True, return_tensors='pt')
        device = torch.device('cuda')
        model.to(device)
        tokens.to(device)

        # Get the model's output for the tokenized text
        output = model(**tokens)

        # Apply softmax to the logits output tensor of our model (in index 0) across dimension -1
        probs = F.softmax(output[0], dim=-1)

        # Get the index of the predicted sentiment class
        pred_class_idx = torch.argmax(probs, dim=1)

        # Map the predicted sentiment class to a sentiment score and interpretation
        sentiment_score = sentiment_scores[pred_class_idx.item()]
        interpretation = sentiment_classes[pred_class_idx.item()]

        # Add the sentiment score and interpretation to the list of results
        results_doc.append({'file_name': filename, 'sentiment_score': sentiment_score, 'interpretation': interpretation})

 # Create a dataframe with the sentiment score and interpretation for each file
df_results_doc = pd.DataFrame(results_doc)

# Print the results dataframe
df = df_results_doc.sort_values('file_name', ascending=False)
df.to_csv('~/GraLNA/New/LLM/sentiment_FinBERT.csv')
