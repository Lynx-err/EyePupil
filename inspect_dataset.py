import zipfile
import pandas as pd

zip_path = "C:/Users/athar/Downloads/eyedentifySR.zip"
sr_method = "RealEsr_x2"

with zipfile.ZipFile(zip_path, 'r') as z:
    # Get first few CSV files
    csv_files = [f for f in z.namelist() 
                if f.startswith(f'eyedentify++/{sr_method}/') 
                and f.endswith('session_data.csv')]
    
    print(f"Found {len(csv_files)} CSV files")
    
    if csv_files:
        print("\nFirst CSV file:", csv_files[0])
        try:
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)
            print("Columns:", list(df.columns))
            print("Shape:", df.shape)
            print("First few rows:")
            print(df.head())
        except Exception as e:
            print(f"Error reading CSV: {e}")
