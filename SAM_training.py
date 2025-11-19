
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import gc
import matplotlib.pyplot as plt
import random


class SAMDecoder(nn.Module):
    def __init__(self, in_channels=256, num_classes=6):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Conv2d(64, num_classes, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.upsample(x1)
        x3 = self.layer2(x2)
        x3 = self.upsample(x3)
        x4 = self.layer3(x3)
        return x4

class KidneySAMTrainer:
    def __init__(self, num_classes, device, sam_model):
        self.device = device
        self.model = sam_model.to(device)
        self.num_classes = num_classes
        
        for name, param in self.model.image_encoder.named_parameters():
            if "blocks" in name:
                # SAM ViT has 12 transformer blocks: blocks.0 ... blocks.11
                # We unfreeze the last 3 (blocks.9, blocks.10, blocks.11)
                block_num = int(name.split('.')[1]) if '.' in name else -1
                if block_num >= 9:  
                    param.requires_grad = True  # trainable
                else:
                    param.requires_grad = False  # frozen
            else:
                # Keep patch embedding & positional encodings frozen
                param.requires_grad = False
        # Enhanced classification head
        self.decoder = SAMDecoder(in_channels=256, num_classes=num_classes).to(device)
        
        encoder_params = [p for p in self.model.image_encoder.parameters() if p.requires_grad]
        decoder_params = list(self.decoder.parameters())
        # Optimizer includes both SAM + head parameters
        self.optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': 1e-5},   # smaller LR for fine-tuning SAM
            {'params': decoder_params, 'lr': 1e-4}    # larger LR for decoder
        ])

        # CrossEntropy for multi-class segmentation
        self.criterion = nn.CrossEntropyLoss()
        
        # Track best IoU instead of loss
        self.best_iou = 0.0
        self.best_dice = 0.0

        # Create visualization directory
        os.makedirs('prediction_visualizations_10x', exist_ok=True)

    def calculate_iou(self, pred, target):
        """Calculate mean IoU - EXCLUDE BACKGROUND (class 0), only classes 1-5"""
        pred = torch.argmax(pred, dim=1)
        ious = []

        # ONLY classes 1-5 (kidney structures), EXCLUDE class 0 (background)
        for cls in range(1, self.num_classes):  # 1,2,3,4,5 - NO class 0
            pred_inds = (pred == cls)
            target_inds = (target == cls)

            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()

            if union == 0:
                # If no ground truth and no prediction, skip this class
                continue
            else:
                ious.append((intersection / union).item())

        # Return mean IoU across kidney structures only
        return np.mean(ious) if len(ious) > 0 else 0.0

    def calculate_dice(self, pred, target):
        """Calculate Dice coefficient - EXCLUDE BACKGROUND (class 0), only classes 1-5"""
        pred = torch.argmax(pred, dim=1)
        dice_scores = []

        # ONLY classes 1-5 (kidney structures), EXCLUDE class 0 (background)
        for cls in range(1, self.num_classes):  # 1,2,3,4,5 - NO class 0
            pred_inds = (pred == cls)
            target_inds = (target == cls)

            intersection = (pred_inds & target_inds).sum().float()
            dice = (2. * intersection) / (pred_inds.sum() + target_inds.sum() + 1e-6)
            dice_scores.append(dice.item())

        return np.mean(dice_scores) if len(dice_scores) > 0 else 0.0

    def calculate_per_class_dice(self, pred, target):
        """Calculate Dice coefficient for each class (1-5) individually"""
        pred = torch.argmax(pred, dim=1)
        class_dice = {}
        
        class_names = {
            1: "Glomeruli",
            2: "Sclerotic", 
            3: "Artery",
            4: "Tubule",
            5: "IFTA"
        }

        for cls in range(1, self.num_classes):  # Only classes 1-5
            pred_inds = (pred == cls)
            target_inds = (target == cls)

            intersection = (pred_inds & target_inds).sum().float()
            dice = (2. * intersection) / (pred_inds.sum() + target_inds.sum() + 1e-6)
            class_dice[class_names[cls]] = dice.item()
            
        return class_dice

    def validate(self, val_loader):
        """Validation phase - compute metrics on validation data"""
        self.model.eval()
        total_loss = 0
        total_iou = 0
        total_dice = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Forward pass
                images = images.float()
                image_embeddings = self.model.image_encoder(images)
                logits = self.decoder(image_embeddings)
                logits_up = nn.functional.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)

                loss = self.criterion(logits_up, masks.long())
                total_loss += loss.item()
                
                # Calculate metrics on VALIDATION data (EXCLUDING background)
                iou = self.calculate_iou(logits_up, masks)  # Only classes 1-5
                dice = self.calculate_dice(logits_up, masks)  # Only classes 1-5
                total_iou += iou
                total_dice += dice

        avg_loss = total_loss / len(val_loader)
        avg_iou = total_iou / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        
        return avg_loss, avg_iou, avg_dice

    def visualize_prediction(self, val_loader, epoch):
        """Visualize prediction for a TRULY random validation image and save it"""
        self.model.eval()
        
        # Get the entire validation dataset
        dataset = val_loader.dataset
        
        # Pick a TRULY random image from the entire validation set
        random_idx = random.randint(0, len(dataset) - 1)
        
        # Load the specific random image and mask
        image, true_mask = dataset[random_idx]
        
        # Add batch dimension
        image = image.unsqueeze(0).to(self.device)
        true_mask = true_mask.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get prediction
            image_embeddings = self.model.image_encoder(image.float())
            logits = self.decoder(image_embeddings)
            pred_mask = torch.argmax(logits, dim=1)
            pred_mask = nn.functional.interpolate(
                pred_mask.unsqueeze(1).float(), 
                size=true_mask.shape[1:], 
                mode='nearest'
            ).squeeze(1).long()
        
        # Convert to numpy for visualization
        image_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        
        true_mask_np = true_mask.squeeze().cpu().numpy()
        pred_mask_np = pred_mask.squeeze().cpu().numpy()
        
        # Define colors for each class
        class_colors = {
            0: [0, 0, 0],        # Background - Black
            1: [1, 0, 0],        # Glomeruli - Red
            2: [0, 1, 0],        # Sclerotic - Green
            3: [0, 0, 1],        # Artery - Blue
            4: [1, 1, 0],        # Tubule - Yellow
            5: [1, 0, 1]         # IFTA - Magenta
        }
        
        # Create colored masks
        true_mask_color = np.zeros((*true_mask_np.shape, 3))
        pred_mask_color = np.zeros((*pred_mask_np.shape, 3))
        
        for class_id, color in class_colors.items():
            true_mask_color[true_mask_np == class_id] = color
            pred_mask_color[pred_mask_np == class_id] = color
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # True mask
        axes[1].imshow(true_mask_color)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(pred_mask_color)
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        # Add legend
        class_names = {
            1: "Glomeruli (Red)",
            2: "Sclerotic (Green)", 
            3: "Artery (Blue)",
            4: "Tubule (Yellow)",
            5: "IFTA (Magenta)"
        }
        
        legend_text = "\n".join([f"{name}" for name in class_names.values()])
        plt.figtext(0.5, 0.01, f"Classes:\n{legend_text}", 
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8})
        
        plt.tight_layout()
        plt.savefig(f'prediction_visualizations_10x/epoch_{epoch:04d}_prediction.png', 
                dpi=150, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        print(f"   📊 Visualization saved: prediction_visualizations_10x/epoch_{epoch:04d}_prediction.png")
        print(f"   🎯 Random validation sample: {random_idx+1}/{len(dataset)}")

    def save_best_model(self, epoch, val_iou, val_dice, val_loss, val_loader):
        """Save ONLY the best model based on VALIDATION IoU and print per-class Dice"""
        # Calculate per-class Dice scores
        self.model.eval()
        per_class_dice = {cls: [] for cls in range(1, self.num_classes)}
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                images = images.float()
                image_embeddings = self.model.image_encoder(images)
                logits = self.decoder(image_embeddings)
                logits_up = nn.functional.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
                
                # Calculate per-class Dice for this batch
                batch_per_class = self.calculate_per_class_dice(logits_up, masks)
                for cls_name, dice_score in batch_per_class.items():
                    # Convert class name back to index for aggregation
                    cls_idx = {
                        "Glomeruli": 1, "Sclerotic": 2, "Artery": 3, 
                        "Tubule": 4, "IFTA": 5
                    }[cls_name]
                    per_class_dice[cls_idx].append(dice_score)
        
        # Average per-class Dice across all batches
        avg_per_class_dice = {}
        class_names = {
            1: "Glomeruli",
            2: "Sclerotic", 
            3: "Artery",
            4: "Tubule", 
            5: "IFTA"
        }
        
        for cls in range(1, self.num_classes):
            if per_class_dice[cls]:
                avg_per_class_dice[class_names[cls]] = np.mean(per_class_dice[cls])
            else:
                avg_per_class_dice[class_names[cls]] = 0.0
        
        # Print per-class Dice scores
        print(f"   📈 Per-class Dice scores:")
        for cls_name, dice_score in avg_per_class_dice.items():
            print(f"      {cls_name}: {dice_score:.4f}")
        
        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_iou': val_iou,      # ← VALIDATION IoU
            'best_dice': val_dice,    # ← VALIDATION Dice  
            'val_loss': val_loss,     # ← VALIDATION loss
            'per_class_dice': avg_per_class_dice,
            'num_classes': self.num_classes
        }
        
        # Save only one file - the best model
        torch.save(checkpoint, 'best_kidney_sam_model_new_10X.pth')
        print(f"   💾 Saved best model - VALIDATION IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        
        # Generate and save visualization
        self.visualize_prediction(val_loader, epoch)

class KidneyDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=1024, is_training=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.is_training = is_training
        
        # Get sorted file lists
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                 if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        # Verify matching files
        assert len(self.image_files) == len(self.mask_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
        
        # Verify file correspondence
        for img_file, mask_file in zip(self.image_files, self.mask_files):
            img_name = os.path.splitext(img_file)[0]
            mask_name = os.path.splitext(mask_file)[0]
            assert img_name == mask_name, f"File mismatch: {img_file} vs {mask_file}"
        
        print(f"Found {len(self.image_files)} images and masks in {images_dir}")

        # SAM-specific preprocessing (same as original SAM)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        # Training augmentations
        if self.is_training:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(degrees=30),
            ])
        else:
            self.augmentation = transforms.Compose([])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and mask paths
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # grayscale segmentation mask
        
        # Verify matching size
        assert image.size == mask.size, f"Size mismatch: {image.size} vs {mask.size}"
        
        # RESIZE to 1024x1024 if not already (your data is already 1024x1024, but this ensures consistency)
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Convert to numpy for augmentation
        image = np.array(image)
        mask = np.array(mask)
        
        # --- Basic augmentations ---
        if self.is_training and np.random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if self.is_training and np.random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        
        # Convert back to PIL for normalization
        image = Image.fromarray(image)
        
        # Apply SAM normalization (ImageNet stats)
        image = self.transform(image)
        
        # Convert mask to tensor (long type for class indices)
        mask = torch.from_numpy(mask).long()
        
        # Verify class range
        unique_vals = torch.unique(mask)
        assert torch.all(unique_vals < 6), f"Invalid mask values: {unique_vals}"
        
        return image, mask

def create_data_loaders(batch_size=2, num_workers=2):
    """Create data loaders for your specific directory structure"""
    
    # Your exact paths
    train_images_path = r"/rodata/dlmpfl/m300305/KIDNEY/NewDataset_10x/Train/image"
    train_masks_path = r"/rodata/dlmpfl/m300305/KIDNEY/NewDataset_10x/Train/mask"
    val_images_path = r"/rodata/dlmpfl/m300305/KIDNEY/NewDataset_10x/Validation/image"
    val_masks_path = r"/rodata/dlmpfl/m300305/KIDNEY/NewDataset_10x/Validation/mask"


    
    # Create datasets
    train_dataset = KidneyDataset(
        images_dir=train_images_path,
        masks_dir=train_masks_path,
        image_size=1024,
        is_training=True
    )
    
    val_dataset = KidneyDataset(
        images_dir=val_images_path,
        masks_dir=val_masks_path, 
        image_size=1024,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # === Data loaders ===
    train_loader, val_loader = create_data_loaders(batch_size=2, num_workers=2)

    # === Initialize SAM ===
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    from segment_anything import sam_model_registry
    sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    print("✅ Loaded pre-trained SAM weights")

    # === Trainer ===
    trainer = KidneySAMTrainer(
        num_classes=6,
        device=device,
        sam_model=sam_model
    )

    # === Resume training if checkpoint exists ===
    resume_path = "best_kidney_sam_model_new_10X.pth"  # Changed to match your save filename
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.best_iou = checkpoint.get("best_iou", 0.0)
        trainer.best_dice = checkpoint.get("best_dice", 0.0)
        print(f"🔄 Resumed from epoch {checkpoint['epoch']} "
              f"(VALIDATION IoU={checkpoint['best_iou']:.4f}, Dice={checkpoint['best_dice']:.4f})")
    else:
        print("🚀 Starting fresh training")

    # === Train with early stopping ===
    epochs = 1000
    patience = 30
    best_val_iou = trainer.best_iou
    counter = 0

    print("Starting training loop with early stopping...")
    print("NOTE: IoU and Dice calculated ONLY for kidney structures (classes 1-5), EXCLUDING background")
    print("Classes: 1=Glomeruli, 2=Sclerotic, 3=Artery, 4=Tubule, 5=IFTA")
    print("Epoch | Train Loss | Train IoU | Train Dice | Val Loss | Val IoU | Val Dice | Status")
    print("-" * 90)
    
    for epoch in range(epochs):
        # === TRAINING PHASE ===
        trainer.model.train()
        train_total_loss = 0
        train_total_iou = 0
        train_total_dice = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            images = images.float()
            image_embeddings = trainer.model.image_encoder(images)
            logits = trainer.decoder(image_embeddings)
            logits_up = nn.functional.interpolate(
                logits, size=masks.shape[1:], mode='bilinear', align_corners=False
            )
            loss = trainer.criterion(logits_up, masks.long())
            train_total_loss += loss.item()

            # Calculate training metrics (EXCLUDING background)
            iou = trainer.calculate_iou(logits_up, masks)  # Only classes 1-5
            dice = trainer.calculate_dice(logits_up, masks)  # Only classes 1-5
            train_total_iou += iou
            train_total_dice += dice

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()
        
        train_avg_loss = train_total_loss / len(train_loader)
        train_avg_iou = train_total_iou / len(train_loader)
        train_avg_dice = train_total_dice / len(train_loader)

        # === VALIDATION PHASE ===
        val_avg_loss, val_avg_iou, val_avg_dice = trainer.validate(val_loader)
        
        # Print both training and validation metrics
        status = ""
        if val_avg_iou > best_val_iou:
            status = "💾 Best"
            best_val_iou = val_avg_iou
            counter = 0
            trainer.best_iou = val_avg_iou      # ← SAVE VALIDATION IoU
            trainer.best_dice = val_avg_dice    # ← SAVE VALIDATION Dice
            # Save model based on VALIDATION metrics (with per-class Dice and visualization)
            trainer.save_best_model(epoch + 1, val_avg_iou, val_avg_dice, val_avg_loss, val_loader)
        else:
            counter += 1
            status = f"⏳ {counter}/{patience}"
            if counter >= patience:
                status = "🛑 Stop"
        
        print(f"{epoch+1:5d} | {train_avg_loss:10.4f} | {train_avg_iou:9.4f} | {train_avg_dice:9.4f} | "
              f"{val_avg_loss:8.4f} | {val_avg_iou:7.4f} | {val_avg_dice:8.4f} | {status}")
        
        if counter >= patience:
            print(f"⏹ Early stopping at epoch {epoch+1}")
            break

    print(f"✅ Training finished. Best VALIDATION IoU: {trainer.best_iou:.4f}, Dice: {trainer.best_dice:.4f}")
    print(f"   (Metrics calculated ONLY for kidney structures: 1=Glomeruli, 2=Sclerotic, 3=Artery, 4=Tubule, 5=IFTA)")

if __name__ == "__main__":
    main()