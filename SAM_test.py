import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import gc
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry

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

class KidneyDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=1024):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        
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

        # SAM-specific preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and mask paths
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Verify matching size
        assert image.size == mask.size, f"Size mismatch: {image.size} vs {mask.size}"
        
        # Resize to 1024x1024 if needed
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Convert to numpy
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Convert back to PIL for transformation
        image_pil = Image.fromarray(image_np)
        
        # Apply SAM normalization
        image_tensor = self.transform(image_pil)
        
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(mask_np).long()
        
        return image_tensor, mask_tensor, self.image_files[idx]

def load_model(checkpoint_path, device):
    """Load the trained model and decoder"""
    # Initialize SAM
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    
    # Initialize decoder
    decoder = SAMDecoder(in_channels=256, num_classes=6)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load state dicts
    sam_model.load_state_dict(checkpoint["model_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    
    # Set to eval mode
    sam_model.to(device).eval()
    decoder.to(device).eval()
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✅ Best Validation IoU: {checkpoint['best_iou']:.4f}, Dice: {checkpoint['best_dice']:.4f}")
    
    return sam_model, decoder

def create_color_mask(mask_np, class_colors):
    """Convert mask to colorful image using class colors"""
    mask_color = np.zeros((*mask_np.shape, 3))
    for class_id, color in class_colors.items():
        mask_color[mask_np == class_id] = color
    return mask_color

def calculate_iou(pred, target, num_classes=6):
    """Calculate IoU for a single image (excluding background)"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    ious = []
    for cls in range(1, num_classes):  # Exclude background
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        
        if union == 0:
            continue
        else:
            ious.append(intersection / union)
    
    return np.mean(ious) if len(ious) > 0 else 0.0

def calculate_dice(pred, target, num_classes=6):
    """Calculate Dice for a single image (excluding background)"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    dice_scores = []
    for cls in range(1, num_classes):  # Exclude background
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        
        intersection = np.logical_and(pred_inds, target_inds).sum()
        dice = (2. * intersection) / (pred_inds.sum() + target_inds.sum() + 1e-6)
        dice_scores.append(dice)
    
    return np.mean(dice_scores) if len(dice_scores) > 0 else 0.0

def predict_all_images(model, decoder, val_loader, device, output_dir):
    """Generate predictions for ALL validation images"""
    model.eval()
    decoder.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Class colors
    class_colors = {
        0: [0, 0, 0],        # Background - Black
        1: [1, 0, 0],        # Glomeruli - Red
        2: [0, 1, 0],        # Sclerotic - Green
        3: [0, 0, 1],        # Artery - Blue
        4: [1, 1, 0],        # Tubule - Yellow
        5: [1, 0, 1]         # IFTA - Magenta
    }
    
    total_images = len(val_loader.dataset)
    all_ious = []
    all_dices = []
    
    print(f"🎯 Processing ALL {total_images} validation images...")
    
    with torch.no_grad():
        for batch_idx, (images, masks, filenames) in enumerate(val_loader):
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            image_embeddings = model.image_encoder(images.float())
            logits = decoder(image_embeddings)
            pred_masks = torch.argmax(logits, dim=1)
            pred_masks = nn.functional.interpolate(
                pred_masks.unsqueeze(1).float(), 
                size=masks.shape[1:], 
                mode='nearest'
            ).squeeze(1).long()
            
            # Process each image in the batch
            for i in range(images.shape[0]):
                # Get individual image data
                image = images[i]
                true_mask = masks[i]
                pred_mask = pred_masks[i]
                filename = filenames[i]
                
                # Convert to numpy for visualization
                image_np = image.cpu().permute(1, 2, 0).numpy()
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)
                
                true_mask_np = true_mask.cpu().numpy()
                pred_mask_np = pred_mask.cpu().numpy()
                
                # Create colorful masks
                true_mask_color = create_color_mask(true_mask_np, class_colors)
                pred_mask_color = create_color_mask(pred_mask_np, class_colors)
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Original image
                axes[0].imshow(image_np)
                axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                
                # True mask
                axes[1].imshow(true_mask_color)
                axes[1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
                axes[1].axis('off')
                
                # Predicted mask
                axes[2].imshow(pred_mask_color)
                axes[2].set_title('Predicted Mask', fontsize=14, fontweight='bold')
                axes[2].axis('off')
                
                # Add legend
                legend_text = "\n".join([f"• {name} ({color_name})" for name, color_name in [
                    ("Glomeruli", "Red"), ("Sclerotic", "Green"), ("Artery", "Blue"), 
                    ("Tubule", "Yellow"), ("IFTA", "Magenta")
                ]])
                
                plt.figtext(0.5, 0.01, f"Class Legend:\n{legend_text}", 
                           ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.9, "pad":10})
                
                # Add filename to title
                base_filename = os.path.splitext(filename)[0]
                plt.suptitle(f'Kidney Structure Segmentation - {base_filename}', fontsize=16, y=0.95)
                
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.15)
                
                # Save figure
                output_path = os.path.join(output_dir, f'prediction_{base_filename}.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
                plt.close()
                
                # Calculate metrics for this image
                iou = calculate_iou(pred_mask.unsqueeze(0), true_mask.unsqueeze(0))
                dice = calculate_dice(pred_mask.unsqueeze(0), true_mask.unsqueeze(0))
                all_ious.append(iou)
                all_dices.append(dice)
                
                print(f"✅ [{batch_idx * val_loader.batch_size + i + 1}/{total_images}] Saved: {output_path} - IoU: {iou:.4f}, Dice: {dice:.4f}")
    
    # Calculate overall metrics
    avg_iou = np.mean(all_ious)
    avg_dice = np.mean(all_dices)
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Average IoU:  {avg_iou:.4f}")
    print(f"   Average Dice: {avg_dice:.4f}")
    print(f"   Total images processed: {total_images}")
    print(f"   All predictions saved to: {output_dir}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    checkpoint_path = "best_kidney_sam_model_new_10X.pth"
    val_images_path = r"C:\Users\m300305\Desktop\tileGen\tile_generation\kidney\Segrenal_paper_dataset\Validation\image"
    val_masks_path = r"C:\Users\m300305\Desktop\tileGen\tile_generation\kidney\Segrenal_paper_dataset\Validation\mask"
    output_dir = "all_validation_predictions_10x"
    
    # Create validation dataset and loader
    val_dataset = KidneyDataset(
        images_dir=val_images_path,
        masks_dir=val_masks_path,
        image_size=1024
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,  # You can adjust batch size based on your GPU memory
        shuffle=False,  # Keep original order
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load trained model
    print("🔄 Loading trained model...")
    sam_model, decoder = load_model(checkpoint_path, device)
    
    # Generate predictions for ALL validation images
    print("\n🎨 Generating predictions for ALL validation images...")
    predict_all_images(
        model=sam_model,
        decoder=decoder,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()