import zipfile

zip_path = "C:/Users/athar/Downloads/eyedentifySR.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    files = z.namelist()
    print("Files in ZIP:")
    for f in sorted(files)[:20]:  # Show first 20 files
        print(f"  {f}")
    
    print("\nLooking for RealEsr_x2 related files:")
    for f in files:
        if 'RealEsr_x2' in f and ('labels.csv' in f or f.endswith('.csv')):
            print(f"  {f}")
