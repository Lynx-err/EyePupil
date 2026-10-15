# test_custom_images.py - Test model on custom eye images

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_model(checkpoint_path, device):
    """Load the trained model"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_pupil_diameter(model, image_path, device, show_processed=False):
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

def process_single_image(model, image_path, device):
    """Process a single image and display results"""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    result = predict_pupil_diameter(model, image_path, device)
    
    if result['success']:
        print(f"✅ Prediction successful!")
        print(f"   Predicted pupil diameter: {result['prediction']:.3f} mm")
        
        # Display image info
        img = result['original_image']
        print(f"   Image size: {img.size[0]}x{img.size[1]} pixels")
        print(f"   Image mode: {img.mode}")
        
        return result
    else:
        print(f"❌ Prediction failed!")
        print(f"   Error: {result['error']}")
        return None

def process_folder(model, folder_path, device):
    """Process all images in a folder"""
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_files = []
    for ext in supported_formats:
        image_files.extend(Path(folder_path).glob(f'*{ext}'))
        image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No supported image files found in {folder_path}")
        print(f"Supported formats: {', '.join(supported_formats)}")
        return []
    
    print(f"Found {len(image_files)} images to process...")
    
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
        print(f"\nPREDICTION STATISTICS:")
        print(f"Average diameter: {np.mean(predictions):.3f} mm")
        print(f"Min diameter: {np.min(predictions):.3f} mm")
        print(f"Max diameter: {np.max(predictions):.3f} mm")
        print(f"Standard deviation: {np.std(predictions):.3f} mm")
        
        print(f"\nINDIVIDUAL RESULTS:")
        print(f"{'Image':<30} {'Prediction (mm)':<15}")
        print(f"{'-'*45}")
        for result in successful:
            filename = Path(result['image_path']).name
            print(f"{filename:<30} {result['prediction']:<15.3f}")
    
    if failed:
        print(f"\nFAILED IMAGES:")
        for result in failed:
            filename = Path(result['image_path']).name
            print(f"  ❌ {filename}: {result['error']}")

def main():
    parser = argparse.ArgumentParser(description='Test model on custom eye images')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to image file or folder containing images')
    parser.add_argument('--model', type=str, default='./checkpoints/best_RealEsr_x2.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of results')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    print(f"Loading model from: {args.model}")
    model = load_model(args.model, device)
    print("✅ Model loaded successfully!")
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        result = process_single_image(model, str(input_path), device)
        results = [result] if result else []
        
    elif input_path.is_dir():
        # Folder of images
        results = process_folder(model, str(input_path), device)
        
    else:
        print(f"❌ Input path not found: {args.input}")
        return
    
    # Print summary
    if results:
        print_summary(results)
        
        # Create visualization if requested
        if args.visualize:
            create_result_visualization(results)
    
    print(f"\n🎉 Processing complete!")

if __name__ == '__main__':
    main()
