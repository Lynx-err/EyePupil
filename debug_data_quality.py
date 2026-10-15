import os
import pandas as pd
import numpy as np
from PIL import Image

data_root = "C:/Users/athar/Datasets/eyedentify++"
sr_method = "RealEsr_x2"

print("=== DEBUGGING DATA QUALITY ===")

# Check pupil diameter values
sr_path = os.path.join(data_root, sr_method)
pupil_diameters = []
bad_files = []

count = 0
for eye_side in ['left_eyes', 'right_eyes']:
    eye_path = os.path.join(sr_path, eye_side)
    if os.path.exists(eye_path):
        for participant in os.listdir(eye_path)[:5]:  # Check first 5 participants
            participant_path = os.path.join(eye_path, participant)
            if os.path.isdir(participant_path):
                for session in os.listdir(participant_path)[:2]:  # First 2 sessions
                    session_path = os.path.join(participant_path, session)
                    csv_file = os.path.join(session_path, 'session_data.csv')
                    
                    if os.path.exists(csv_file):
                        try:
                            df = pd.read_csv(csv_file)
                            diameters = df['pupil_diameter'].values
                            
                            # Check for problematic values
                            nan_count = np.isnan(diameters).sum()
                            inf_count = np.isinf(diameters).sum()
                            zero_count = (diameters == 0).sum()
                            negative_count = (diameters < 0).sum()
                            
                            if nan_count > 0 or inf_count > 0 or negative_count > 0:
                                bad_files.append(csv_file)
                                print(f"BAD FILE: {csv_file}")
                                print(f"  NaN: {nan_count}, Inf: {inf_count}, Negative: {negative_count}, Zero: {zero_count}")
                            
                            pupil_diameters.extend(diameters[~np.isnan(diameters) & ~np.isinf(diameters)])
                            count += len(diameters)
                            
                            if count > 1000:  # Limit sample size
                                break
                        except Exception as e:
                            print(f"Error reading {csv_file}: {e}")
                            bad_files.append(csv_file)
                    
                    if count > 1000:
                        break
                if count > 1000:
                    break
        if count > 1000:
            break

print(f"\n=== PUPIL DIAMETER STATISTICS ===")
if pupil_diameters:
    pupil_diameters = np.array(pupil_diameters)
    print(f"Total samples checked: {len(pupil_diameters)}")
    print(f"Min: {pupil_diameters.min():.4f}")
    print(f"Max: {pupil_diameters.max():.4f}")
    print(f"Mean: {pupil_diameters.mean():.4f}")
    print(f"Std: {pupil_diameters.std():.4f}")
    print(f"Values > 100: {(pupil_diameters > 100).sum()}")
    print(f"Values < 0.1: {(pupil_diameters < 0.1).sum()}")
else:
    print("No valid pupil diameter values found!")

print(f"\n=== BAD FILES ===")
print(f"Found {len(bad_files)} problematic files")

# Check a sample image
print(f"\n=== IMAGE CHECK ===")
sample_img_path = "C:/Users/athar/Datasets/eyedentify++/RealEsr_x2/left_eyes/1/1/frame_01.png"
if os.path.exists(sample_img_path):
    try:
        img = Image.open(sample_img_path)
        print(f"Image mode: {img.mode}")
        print(f"Image size: {img.size}")
        
        # Convert to RGB and check pixel values
        img_rgb = img.convert('RGB')
        import numpy as np
        img_array = np.array(img_rgb)
        print(f"Pixel value range: {img_array.min()} to {img_array.max()}")
        print(f"Image shape: {img_array.shape}")
        
    except Exception as e:
        print(f"Error loading sample image: {e}")
else:
    print(f"Sample image not found: {sample_img_path}")
