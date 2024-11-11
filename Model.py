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
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
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
        
        # Modified loading logic
        if os.path.exists('skin_cancer_model.pth'):
            try:
                state_dict = torch.load('skin_cancer_model.pth')
                self.model.load_state_dict(state_dict)
                logger.info("Loaded pre-trained model weights.")
            except Exception as e:
                logger.warning(f"Could not load previous model weights: {str(e)}")
                logger.info("Training new model from scratch.")

    @staticmethod
    def prepare_data(metadata_path, images_dir):
        metadata = pd.read_csv(metadata_path)
        
        # Filter out missing images
        valid_image_paths = []
        valid_labels = []
        for idx, row in metadata.iterrows():
            image_path = os.path.join(images_dir, f"{row['image_id']}.jpg")
            if os.path.exists(image_path):
                valid_image_paths.append(image_path)
                valid_labels.append(row['dx'])
        
        # Log the number of valid images
        logger.info(f"Found {len(valid_image_paths)} valid images out of {len(metadata)} total entries")
        
        # Check if we have enough data to proceed
        if len(valid_image_paths) == 0:
            raise ValueError("No valid images found in the specified directory")
        
        label_counts = pd.Series(valid_labels).value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")

        return train_test_split(valid_image_paths, valid_labels, test_size=0.2, random_state=42)

    @staticmethod
    def get_transform(train=False):
        if train:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        transform = self.get_transform()
        image_tensor = transform(image).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0))
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
        
        # Create datasets with appropriate transforms
        train_dataset = SkinLesionDataset(train_images, train_labels, transform=cls.get_transform(train=True))
        val_dataset = SkinLesionDataset(val_images, val_labels, transform=cls.get_transform(train=False))
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        # Training setup
        num_epochs = 30  # Increased epochs since we'll use early stopping
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Early stopping setup
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        best_model_state = None

        for epoch in range(num_epochs):
            model.model.train()
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                
                optimizer.zero_grad()
                outputs = model.model(inputs)
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
            val_acc = 100. * correct / total

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.model.state_dict().copy()
            else:
                patience_counter += 1

            scheduler.step()
            
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, '
                       f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after epoch {epoch+1}')
                break

        # Load best model state
        model.model.load_state_dict(best_model_state)
        torch.save(best_model_state, 'skin_cancer_model.pth')
        logger.info(f"Model trained and saved. Best validation accuracy: {best_val_acc:.2f}%")

    def label_to_index(self, label):
        return list(self.label_dict.keys()).index(label)
