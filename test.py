import pandas as pd

file_path = r'C:\tmp\vishing_detection\src\models\data\training\transcripts.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Success! Loaded {len(df)} rows.")
except PermissionError:
    print("Permission denied - file likely locked by another process.")
except FileNotFoundError:
    print("File not found - check the exact path.")