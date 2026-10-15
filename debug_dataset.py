import os
import pandas as pd

data_root = "C:/Users/athar/Datasets/eyedentify++"
sr_method = "RealEsr_x2"

# Check the directory structure first
sr_path = os.path.join(data_root, sr_method)
print(f"SR path: {sr_path}")
print(f"SR path exists: {os.path.exists(sr_path)}")

if os.path.exists(sr_path):
    print("Contents of SR path:", os.listdir(sr_path))
    
    # Check left_eyes structure
    left_eyes_path = os.path.join(sr_path, "left_eyes")
    if os.path.exists(left_eyes_path):
        participants = os.listdir(left_eyes_path)[:3]  # First 3 participants
        print(f"First few participants: {participants}")
        
        for participant in participants:
            participant_path = os.path.join(left_eyes_path, participant)
            if os.path.isdir(participant_path):
                sessions = os.listdir(participant_path)
                print(f"Participant {participant} sessions: {sessions}")
                
                for session in sessions[:1]:  # First session only
                    session_path = os.path.join(participant_path, session)
                    csv_file = os.path.join(session_path, 'session_data.csv')
                    
                    if os.path.exists(csv_file):
                        print(f"\nFound CSV: {csv_file}")
                        df = pd.read_csv(csv_file)
                        print(f"CSV columns: {list(df.columns)}")
                        print(f"CSV shape: {df.shape}")
                        
                        if len(df) > 0:
                            first_row = df.iloc[0]
                            print(f"First row frame_path: {first_row['frame_path']}")
                            print(f"First row pupil_diameter: {first_row['pupil_diameter']}")
                            
                            # Check if image exists
                            frame_path = first_row['frame_path']
                            img_path = os.path.join(session_path, frame_path)
                            print(f"Full image path: {img_path}")
                            print(f"Image exists: {os.path.exists(img_path)}")
                            
                            # List what's actually in the session directory
                            print(f"Session directory contents: {os.listdir(session_path)}")
                        break
                break
