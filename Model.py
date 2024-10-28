import torch
from torch import nn
from torchvision import transforms
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_to_index = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert string label to numeric index
        label_index = self.label_to_index[label]
        
        return image, torch.tensor(label_index, dtype=torch.long)

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SkinCancerModel:
    def __init__(self):
        self.label_dict = {
            'akiec': "Actinic Keratoses and Bowen's disease",
            'bcc': "Basal Cell Carcinoma",
            'bkl': "Benign Keratosis-like Lesions",
            'df': "Dermatofibroma",
            'mel': "Melanoma",
            'nv': "Melanocytic Nevi",
            'vasc': "Vascular Lesions"
        }
        self.label_to_index = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
        self.model = CustomCNN(num_classes=len(self.label_dict))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Add DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        
        if os.path.exists('skin_cancer_model.pth'):
            self.model.load_state_dict(torch.load('skin_cancer_model.pth'))
            logger.info("Loaded pre-trained model weights.")

    @staticmethod
    def prepare_data(metadata_path, images_dir):
        metadata = pd.read_csv(metadata_path)
        image_paths = [os.path.join(images_dir, f"{image_id}.jpg") for image_id in metadata['image_id']]
        labels = metadata['dx'].values

        # No need to convert labels to numeric indices
        label_counts = pd.Series(labels).value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")

        return train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    @staticmethod
    def get_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect_skin(self, image):
        # Convert the image to numpy array and then to BGR color space
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Convert to YCrCb color space
        image_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

        # Adjust skin color range for medical images
        lower_skin = np.array([0, 125, 75], dtype=np.uint8)  # Relaxed lower bounds
        upper_skin = np.array([255, 188, 145], dtype=np.uint8)  # Increased upper bounds

        # Create binary mask
        skin_mask = cv2.inRange(image_ycrcb, lower_skin, upper_skin)

        # Enhance mask with additional color space check
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin_hsv = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask_hsv = cv2.inRange(image_hsv, lower_skin_hsv, upper_skin_hsv)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(skin_mask, skin_mask_hsv)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Reduced kernel size
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

        # Fill holes in the mask
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        return skin_mask

    def predict(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        transform = self.get_transform()
        image_tensor = transform(image).to(self.device)

        # Detect skin
        skin_mask = self.detect_skin(image_tensor)
        
        # Adjust threshold for skin detection (lower threshold for medical images)
        if np.sum(skin_mask) / (skin_mask.shape[0] * skin_mask.shape[1]) < 0.01:  # Changed from 0.05 to 0.01
            return "Please upload an image containing human skin.", 0

        # Apply the skin mask to the image
        masked_image = image_tensor * torch.from_numpy(skin_mask).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(masked_image.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Convert predicted index to label
        index_to_label = {v: k for k, v in self.label_to_index.items()}
        predicted_label = index_to_label.get(predicted.item(), "Unknown")
        predicted_label = self.label_dict.get(predicted_label, "Unknown")
        
        # Convert confidence to percentage with 2 decimal places and add '%' symbol
        confidence_percent = f"{(confidence.item() * 100):.2f}%"

        return predicted_label, confidence_percent

    @classmethod
    def train_with_ham10000(cls, metadata_path, images_dir):
        model = cls()
        train_images, val_images, train_labels, val_labels = cls.prepare_data(metadata_path, images_dir)
        
        # Create datasets
        train_dataset = SkinLesionDataset(train_images, train_labels, transform=cls.get_transform())
        val_dataset = SkinLesionDataset(val_images, val_labels, transform=cls.get_transform())
        
        # Create data loaders with num_workers
        num_workers = 4  # Adjust based on your CPU cores
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        # Training logic
        num_epochs = 10
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        for epoch in range(num_epochs):
            model.model.train()
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                
                # Apply skin detection (consider moving this to the dataset class for preprocessing)
                skin_masks = torch.stack([torch.from_numpy(model.detect_skin(img)) for img in inputs])
                masked_inputs = inputs * skin_masks.float().unsqueeze(1).to(model.device)
                
                optimizer.zero_grad()
                outputs = model.model(masked_inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    outputs = model.model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {100.*correct/total:.2f}%')

        torch.save(model.model.state_dict(), 'skin_cancer_model.pth')
        logger.info("Model trained and saved.")

    def label_to_index(self, label):
        return list(self.label_dict.keys()).index(label)
