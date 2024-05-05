import pandas as pd

# Load the dataset
file_path = r'C:\Users\rodyj\Documents\data\conn.log.labelled.txt'
try:
    data = pd.read_csv(file_path, sep='\t', skiprows=6)  # Set the separator as '\t' and skip the first 6 rows
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit(1)  # Exit the script if file not found

# Print the first few rows and data types of each column
print(data.head())
print(data.dtypes)
