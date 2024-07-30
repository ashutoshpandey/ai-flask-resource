import pandas as pd

# Load CSV file
def load_csv(file_path):
    data = pd.read_csv(file_path)
    return data

