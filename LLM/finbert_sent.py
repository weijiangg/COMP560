import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

# Function to process sentiment analysis for a given directory
def process_sentiment_for_directory(dir_path, tokenizer, model):
    start_time = time.time()  # Start time for the directory processing
    print(f"Processing directory: {dir_path}")
    
    # Initialize an empty list to store the results
    results_doc = []

    # Define the sentiment classes and scores
    sentiment_classes = ['positive', 'neutral', 'negative']
    sentiment_scores = [1, 0, -1]

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
            
            # Move tokens to CUDA device if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tokens = tokens.to(device)

            # Get the model's output for the tokenized text
            model.to(device)
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

    # Save the results to CSV file
    output_csv = os.path.join(dir_path, 'sentiment_results.csv')
    df_results_doc.to_csv(output_csv, index=False)
    
    end_time = time.time()  # End time for the directory processing
    duration = end_time - start_time  # Duration in seconds
    
    print(f"Finished processing directory: {dir_path}. Time taken: {duration:.2f} seconds")
    print(f"Sentiment results saved to '{output_csv}'")

# Main script to iterate through each year directory
def main():
    # Set the base directory containing the year folders
    base_dir = '/home/r_huang3/GraLNA/New/LLM/annualreport'

    # Load the FinBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

    # Iterate through each year folder
    for year_folder in os.listdir(base_dir):
        year_dir = os.path.join(base_dir, year_folder)
        
        # Ensure it's a directory (excluding potential non-directory files)
        if os.path.isdir(year_dir):
            print(f"Processing year: {year_folder}")
            
            # Process train, test, and validation directories if they exist
            for data_type in ['train', 'test', 'validation']:
                data_dir = os.path.join(year_dir, data_type)
                
                # Check if the data directory exists
                if os.path.exists(data_dir):
                    process_sentiment_for_directory(data_dir, tokenizer, model)
                else:
                    print(f"{data_type} directory does not exist in {year_folder}")

# Entry point of the script
if __name__ == "__main__":
    main()

