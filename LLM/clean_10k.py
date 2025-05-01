import datasets
import pandas as pd
import os

# Load the entire dataset
dataset_dict = datasets.load_dataset("eloukas/edgar-corpus", "full")

# Define the output directory
output_dir = os.path.expanduser("~/New/LLM/annualreport")
os.makedirs(output_dir, exist_ok=True)

# Iterate over each dataset in the DatasetDict
for split, dataset in dataset_dict.items():
    # Convert to pandas DataFrame
    df = dataset.to_pandas()

    # Get the header
    header = df.columns.tolist()

    # Save the header to a separate text file for each split (optional)
    header_path = os.path.join(output_dir, f"header_{split}.txt")
    with open(header_path, "w") as header_file:
        header_file.write(",".join(header) + "\n")

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        first_column_value = str(row.iloc[0])
        year = str(row.iloc[2])  # Assuming the third column contains the year

        # Remove .txt or .htm from the first column value
        first_column_value = first_column_value.replace('.txt', '').replace('.htm', '')

        # Create a folder for the year if it doesn't exist
        year_dir = os.path.join(output_dir, year)
        os.makedirs(year_dir, exist_ok=True)

        # Create subfolders for train, val, and test splits
        split_dir = os.path.join(year_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Define the file path for the row
        row_file_path = os.path.join(split_dir, f"{first_column_value}.txt")

        # Save only the twelfth column to the text file
        with open(row_file_path, "w") as row_file:
            row_file.write(f"{header[11]}\n")  # Write the header for the twelfth column
            row_file.write(f"{row.iloc[11]}\n")  # Write the value of the twelfth column

