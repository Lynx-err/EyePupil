# train_eyedentify_pp_clean.py

import os
import argparse
import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda.amp import GradScaler, autocast
import random
import numpy as np

class EyeDentifyPPDataset(Dataset):
    def __init__(self, data_root, sr_method, split='train', test_size=0.2, 
                 random_state=42, transform=None, eye_type='both', max_samples_per_csv=50):
        self.transform = transform
        self.samples = []
        random.seed(random_state)
        
        sr_path = os.path.join(data_root, sr_method)
        csv_files = []
        
        print(f"Looking for data in: {sr_path}")
        
        for eye_side in ['left_eyes', 'right_eyes']:
            if eye_type != 'both' and ((eye_type == 'left' and eye_side != 'left_eyes') or 
                                       (eye_type == 'right' and eye_side != 'right_eyes')):
                continue
            
            eye_path = os.path.join(sr_path, eye_side)
            
            if os.path.exists(eye_path):
                for participant in os.listdir(eye_path):
                    participant_path = os.path.join(eye_path, participant)
                    if os.path.isdir(participant_path):
                        for session in os.listdir(participant_path):
                            session_path = os.path.join(participant_path, session)
                            csv_file = os.path.join(session_path, 'session_data.csv')
                            if os.path.exists(csv_file):
                                csv_files.append((csv_file, eye_side, participant, session))
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Process CSV files with sample limit and NaN filtering
        nan_count = 0
        for csv_file, eye_side, participant_id, session_id in csv_files:
            try:
                df = pd.read_csv(csv_file)
                sample_count = 0
                
                for _, row in df.iterrows():
                    if sample_count >= max_samples_per_csv:
                        break
                    
                    pupil_diameter = row['pupil_diameter']
                    
                    # CRITICAL FIX: Skip NaN, infinite, or unrealistic values
                    if pd.isna(pupil_diameter) or np.isinf(pupil_diameter) or pupil_diameter <= 0:
                        nan_count += 1
                        continue
                    
                    frame_path = row['frame_path']  # e.g., "1/1/frame_01.png"
                    
                    # Extract just the filename from frame_path
                    frame_filename = os.path.basename(frame_path)  # Gets "frame_01.png"
                    
                    # Build correct image path - images are directly in session directory
                    img_path = os.path.join(os.path.dirname(csv_file), frame_filename)
                    
                    if os.path.exists(img_path):
                        self.samples.append({
                            'img_path': img_path,
                            'pupil_diameter': float(pupil_diameter),
                            'eye_side': eye_side,
                            'participant': participant_id,
                            'session': session_id
                        })
                        sample_count += 1
            
            except Exception as e:
                print(f"Warning: Could not process {csv_file}: {e}")
                continue
        
        print(f"Total samples found: {len(self.samples)}")
        print(f"Skipped {nan_count} samples with NaN/invalid pupil diameter values")
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples found! Check the dataset structure.")
        
        # Train/val split
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * (1 - test_size))
        
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"Loaded {len(self.samples)} samples for {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        diameter = sample['pupil_diameter']
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            img = Image.new('RGB', (64, 128), color='black')
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor([diameter], dtype=torch.float32)

def get_data_loaders(data_root, sr_method, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    
    train_ds = EyeDentifyPPDataset(data_root, sr_method, 'train', transform=transform, max_samples_per_csv=50)
    val_ds = EyeDentifyPPDataset(data_root, sr_method, 'val', transform=transform, max_samples_per_csv=50)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def build_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, targets).item() * images.size(0)
    return total_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on EyeDentify++ Dataset')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--sr-method', type=str, default='RealEsr_x2')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=str, default='./checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_data_loaders(
        args.data_root, args.sr_method, args.batch_size, args.workers)

    model = build_model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    os.makedirs(args.output, exist_ok=True)
    best_val = float('inf')
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion,
                                  optimizer, scaler, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch}/{args.epochs}  '
              f'Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.output, f'best_{args.sr_method}.pt')
            torch.save(model.state_dict(), ckpt)
            print(f'  *** Saved best model to {ckpt} ***')

if __name__ == '__main__':
    main()
