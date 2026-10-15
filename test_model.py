# test_model.py - Simple model testing script

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
import numpy as np

def load_model(checkpoint_path, device):
    """Load the trained model"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_pupil_diameter(model, image_path, device):
    """Predict pupil diameter for a single image"""
    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(image_tensor)
        
        return prediction.item()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def test_sample_images(model, data_root, sr_method, device, num_samples=20):
    """Test model on sample images with ground truth"""
    results = []
    
    # Get some sample images from the dataset
    sr_path = os.path.join(data_root, sr_method)
    
    count = 0
    for eye_side in ['left_eyes', 'right_eyes']:
        eye_path = os.path.join(sr_path, eye_side)
        if os.path.exists(eye_path):
            participants = os.listdir(eye_path)[:5]  # First 5 participants
            
            for participant in participants:
                if count >= num_samples:
                    break
                    
                participant_path = os.path.join(eye_path, participant)
                if os.path.isdir(participant_path):
                    sessions = os.listdir(participant_path)[:2]  # First 2 sessions
                    
                    for session in sessions:
                        if count >= num_samples:
                            break
                            
                        session_path = os.path.join(participant_path, session)
                        csv_file = os.path.join(session_path, 'session_data.csv')
                        
                        if os.path.exists(csv_file):
                            try:
                                df = pd.read_csv(csv_file)
                                
                                # Test first few frames from this session
                                for i, row in df.head(2).iterrows():
                                    if count >= num_samples:
                                        break
                                        
                                    pupil_diameter = row['pupil_diameter']
                                    
                                    # Skip invalid values
                                    if pd.isna(pupil_diameter) or pupil_diameter <= 0:
                                        continue
                                    
                                    frame_path = row['frame_path']
                                    frame_filename = os.path.basename(frame_path)
                                    img_path = os.path.join(session_path, frame_filename)
                                    
                                    if os.path.exists(img_path):
                                        prediction = predict_pupil_diameter(model, img_path, device)
                                        
                                        if prediction is not None:
                                            error = abs(prediction - pupil_diameter)
                                            relative_error = (error / pupil_diameter) * 100
                                            
                                            results.append({
                                                'image': frame_filename,
                                                'eye_side': eye_side,
                                                'participant': participant,
                                                'ground_truth': pupil_diameter,
                                                'prediction': prediction,
                                                'absolute_error': error,
                                                'relative_error_%': relative_error
                                            })
                                            
                                            count += 1
                                            
                            except Exception as e:
                                print(f"Error processing {csv_file}: {e}")
                
                if count >= num_samples:
                    break
            
            if count >= num_samples:
                break
    
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    checkpoint_path = './checkpoints/best_RealEsr_x2.pt'
    data_root = "C:/Users/athar/Datasets/eyedentify++"
    sr_method = "RealEsr_x2"
    
    # Check if model exists
    if not os.path.exists(checkpoint_path):
        print(f"Model not found: {checkpoint_path}")
        return
    
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    print("Model loaded successfully! ✅")
    
    # Test on sample images
    print("\n=== TESTING MODEL ON SAMPLE IMAGES ===")
    results = test_sample_images(model, data_root, sr_method, device, num_samples=20)
    
    if results:
        print(f"\nTested {len(results)} images:")
        print("-" * 90)
        print(f"{'Image':<20} {'Eye Side':<10} {'Ground Truth':<12} {'Prediction':<12} {'Error':<8} {'Rel Error %':<10}")
        print("-" * 90)
        
        total_error = 0
        for result in results:
            print(f"{result['image']:<20} {result['eye_side']:<10} "
                  f"{result['ground_truth']:<12.3f} {result['prediction']:<12.3f} "
                  f"{result['absolute_error']:<8.3f} {result['relative_error_%']:<10.1f}")
            total_error += result['absolute_error']
        
        # Calculate summary statistics
        errors = [r['absolute_error'] for r in results]
        rel_errors = [r['relative_error_%'] for r in results]
        
        print("-" * 90)
        print(f"SUMMARY STATISTICS:")
        print(f"Average Error:        {np.mean(errors):.3f} mm")
        print(f"Max Error:           {np.max(errors):.3f} mm")
        print(f"Min Error:           {np.min(errors):.3f} mm")
        print(f"Standard Deviation:  {np.std(errors):.3f} mm")
        print(f"Average Relative Error: {np.mean(rel_errors):.1f}%")
        
        if np.mean(errors) < 0.1:
            print("🎉 Excellent accuracy! Your model is performing very well!")
        elif np.mean(errors) < 0.2:
            print("👍 Very good accuracy! Your model is working well!")
        elif np.mean(errors) < 0.3:
            print("👌 Good accuracy! Acceptable performance!")
        else:
            print("⚠️  Model needs improvement. Consider more training or data.")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv('model_test_results.csv', index=False)
        print(f"\nResults saved to: model_test_results.csv")
        
    else:
        print("No test results obtained. Check your dataset path and structure.")

if __name__ == '__main__':
    main()
