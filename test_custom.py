# test_custom_images.py - Test model on images/ folder automatically

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_model(checkpoint_path, device):
    """Load the trained model"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_pupil_diameter(model, image_path, device):
    """Predict pupil diameter for a custom image"""
    
    # Same preprocessing as training
    transform = transforms.Compose([
        transforms.Resize((64, 128)),  # Resize to training dimensions
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    
    try:
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        processed_image = original_image.resize((128, 64))  # For display
        
        # Make prediction
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(image_tensor)
        
        result = {
            'image_path': image_path,
            'original_image': original_image,
            'processed_image': processed_image,
            'prediction': prediction.item(),
            'success': True,
            'error': None
        }
        
        return result
        
    except Exception as e:
        return {
            'image_path': image_path,
            'original_image': None,
            'processed_image': None,
            'prediction': None,
            'success': False,
            'error': str(e)
        }

def create_result_visualization(results, save_path='custom_results.png'):
    """Create a visualization showing original images and predictions"""
    
    successful_results = [r for r in results if r['success']]
    if not successful_results:
        print("No successful predictions to visualize")
        return
    
    n_images = len(successful_results)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    if n_images == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, result in enumerate(successful_results):
        if i >= len(axes):
            break
            
        # Display original image
        axes[i].imshow(result['original_image'])
        axes[i].set_title(f"{Path(result['image_path']).name}\n"
                         f"Predicted: {result['prediction']:.3f}mm", 
                         fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(successful_results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Results visualization saved to: {save_path}")

def process_images_folder(model, folder_path, device):
    """Process all images in the specified folder"""
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        print("Please create an 'images' folder in your EyePupil directory")
        return []
    
    image_files = []
    for ext in supported_formats:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No supported image files found in {folder_path}")
        print(f"Supported formats: JPG, JPEG, PNG, BMP, TIFF")
        print("Please add eye images to the folder and try again")
        return []
    
    print(f"✅ Found {len(image_files)} images in {folder_path}")
    print(f"Supported formats: JPG, JPEG, PNG, BMP, TIFF")
    
    results = []
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file.name}")
        
        result = predict_pupil_diameter(model, str(image_file), device)
        results.append(result)
        
        if result['success']:
            print(f"  ✅ Predicted diameter: {result['prediction']:.3f} mm")
        else:
            print(f"  ❌ Error: {result['error']}")
    
    return results

def print_summary(results):
    """Print summary statistics"""
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(results)}")
    print(f"Successful predictions: {len(successful)}")
    print(f"Failed predictions: {len(failed)}")
    
    if successful:
        predictions = [r['prediction'] for r in successful]
        print(f"\n📊 PREDICTION STATISTICS:")
        print(f"Average diameter:    {np.mean(predictions):.3f} mm")
        print(f"Min diameter:        {np.min(predictions):.3f} mm")
        print(f"Max diameter:        {np.max(predictions):.3f} mm")
        print(f"Standard deviation:  {np.std(predictions):.3f} mm")
        
        print(f"\n📋 DETAILED RESULTS:")
        print(f"{'Image':<30} {'Prediction (mm)':<15}")
        print(f"{'-'*45}")
        for result in successful:
            filename = Path(result['image_path']).name
            print(f"{filename:<30} {result['prediction']:<15.3f}")
    
    if failed:
        print(f"\n❌ FAILED IMAGES:")
        for result in failed:
            filename = Path(result['image_path']).name
            print(f"  {filename}: {result['error']}")

def main():
    """Main function - automatically processes ./images/ folder"""
    
    # Default paths for EyePupil project
    script_dir = Path(__file__).parent  # EyePupil directory
    images_folder = script_dir / "images"
    model_path = script_dir / "checkpoints" / "best_RealEsr_x2.pt"
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🤖 Pupil Diameter Tester")
    print(f"Using device: {device}")
    print(f"Images folder: {images_folder}")
    print(f"Model: {model_path}")
    print("-" * 60)
    
    # Check model exists
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please ensure './checkpoints/best_RealEsr_x2.pt' exists")
        return
    
    # Load model
    print(f"Loading model...")
    model = load_model(str(model_path), device)
    print("✅ Model loaded successfully!")
    
    # Process images folder
    results = process_images_folder(model, images_folder, device)
    
    # Print summary
    if results:
        print_summary(results)
        
        # Auto-generate visualization
        create_result_visualization(results)
        
        # Save results to CSV
        successful_results = [r for r in results if r['success']]
        if successful_results:
            df = pd.DataFrame([{
                'filename': Path(r['image_path']).name,
                'prediction_mm': r['prediction'],
                'success': r['success']
            } for r in successful_results])
            df.to_csv('pupil_measurements.csv', index=False)
            print(f"\n💾 Results saved to: pupil_measurements.csv")
    
    print(f"\n🎉 Analysis complete! Check 'custom_results.png' for visualization.")

if __name__ == '__main__':
    main()
