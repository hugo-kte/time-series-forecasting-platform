import os
import pandas as pd

def load_data(filename="full_data.csv"):
    current_dir = os.path.dirname(__file__)            # backend/
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # racine projet
    data_path = os.path.join(parent_dir, "data", filename)

    df = pd.read_csv(data_path, parse_dates=["Datetime"])
    return df