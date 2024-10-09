import torch  # Import PyTorch library
from torchvision import models, transforms  # Import models and transforms from torchvision
import timm  # Import timm library for pre-trained models
import logging  # Import logging library for logging information
from PIL import Image  # Import Image class from PIL for image processing
from torch.utils.data import Dataset, DataLoader  # Import Dataset and DataLoader for data handling
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data
import pandas as pd  # Import pandas for data manipulation
import os  # Import os for file and directory operations

# Set up logging
logging.basicConfig(level=logging.INFO)  # Configure logging to show INFO level messages
logger = logging.getLogger(__name__)  # Create a logger object

class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # Store image paths
        self.labels = labels  # Store labels
        self.transform = transform  # Store transform function

    def __len__(self):
        return len(self.image_paths)  # Return the number of images

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')  # Open image and convert to RGB
        label = self.labels[idx]  # Get the label for the image

        if self.transform:
            image = self.transform(image)  # Apply transform if provided

        return image, label  # Return the image and its label

class SkinCancerModel:
    def __init__(self):
        self.label_dict = {
            0: "Benign Keratosis-like Lesions",
            1: "Melanocytic Nevi",
            2: "Melanoma",
            3: "Dermatofibroma",
            4: "Vascular Lesions",
            5: "Basal Cell Carcinoma",
            6: "Actinic Keratoses",
            7: "Melanocytic Nevus"
        }  # Dictionary mapping label indices to label names
        
        self.model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=len(self.label_dict))  # Create a pre-trained model with the specified number of classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU
        self.model.to(self.device)  # Move the model to the appropriate device

        if os.path.exists('skin_cancer_model.pth'):
            self.model.load_state_dict(torch.load('skin_cancer_model.pth'))  # Load model weights if the file exists
            logger.info("Loaded pre-trained model weights.")  # Log that the model weights were loaded

    @staticmethod
    def prepare_data(metadata_path, images_dir):
        metadata = pd.read_csv(metadata_path)  # Read metadata from CSV file
        image_paths = [os.path.join(images_dir, f"{image_id}.jpg") for image_id in metadata['image_id']]  # Create list of image paths
        labels = metadata['dx'].map({
            "bkl": 0,
            "nv": 1,
            "mel": 2,
            "df": 3,
            "vasc": 4,
            "bcc": 5,
            "akiec": 6,
            "mns": 7
        }).values  # Map diagnosis labels to numerical values

        # Log the distribution of labels
        label_counts = pd.Series(labels).value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")

        return train_test_split(image_paths, labels, test_size=0.2, random_state=42)  # Split data into training and validation sets

    @staticmethod
    def get_train_transform():
        return transforms.Compose([
            transforms.RandomResizedCrop(224),  # Randomly crop and resize images to 224x224
            transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
            transforms.RandomVerticalFlip(),  # Randomly flip images vertically
            transforms.RandomRotation(20),  # Randomly rotate images by up to 20 degrees
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images with mean and std
        ])  # Return a composed transform for training

    @staticmethod
    def get_val_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images with mean and std
        ])  # Return a composed transform for validation

    @classmethod
    def train_with_ham10000(cls, metadata_path, images_dir):
        model = cls()  # Create an instance of SkinCancerModel
        train_images, val_images, train_labels, val_labels = cls.prepare_data(metadata_path, images_dir)  # Prepare data
        model.train(
            train_data={'image_paths': train_images, 'labels': train_labels},  # Training data
            val_data={'image_paths': val_images, 'labels': val_labels},  # Validation data
            epochs=10,  # Number of epochs
            batch_size=32,  # Batch size
            learning_rate=0.001  # Learning rate
        )  # Train the model

    def train(self, train_data, val_data, epochs=10, batch_size=32, learning_rate=0.001):
        train_dataset = SkinLesionDataset(train_data['image_paths'], train_data['labels'], transform=self.get_train_transform())  # Create training dataset
        val_dataset = SkinLesionDataset(val_data['image_paths'], val_data['labels'], transform=self.get_val_transform())  # Create validation dataset

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Create DataLoader for training data
        val_loader = DataLoader(val_dataset, batch_size=batch_size)  # Create DataLoader for validation data

        criterion = torch.nn.CrossEntropyLoss()  # Define loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # Define optimizer

        for epoch in range(epochs):
            self.model.train()  # Set model to training mode
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)  # Move images and labels to device
                
                optimizer.zero_grad()  # Zero the gradients
                outputs = self.model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

            # Validation
            self.model.eval()  # Set model to evaluation mode
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)  # Move images and labels to device
                    outputs = self.model(images)  # Forward pass
                    val_loss += criterion(outputs, labels).item()  # Compute validation loss
                    _, predicted = outputs.max(1)  # Get predicted labels
                    total += labels.size(0)  # Total number of labels
                    correct += predicted.eq(labels).sum().item()  # Number of correct predictions

            logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100.*correct/total:.2f}%')  # Log training and validation results

        torch.save(self.model.state_dict(), 'skin_cancer_model.pth')  # Save model weights

    def predict(self, image_path):
        self.model.eval()  # Set model to evaluation mode
        image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
        image_tensor = self.get_val_transform()(image).unsqueeze(0).to(self.device)  # Apply validation transform and add batch dimension
        
        with torch.no_grad():
            output = self.model(image_tensor)  # Forward pass
            probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Compute probabilities
            predicted_index = torch.argmax(output, dim=1).item()  # Get predicted label index

        confidence = probabilities[predicted_index].item()  # Get confidence of the prediction
        predicted_label = self.label_dict[predicted_index]  # Get predicted label

        return predicted_label, confidence  # Return predicted label and confidence

    def add_new_data(self, image_path, label):
        """
        Add new data to the training set and retrain the model.
        """
        # Append new data to the existing dataset
        new_image_path = os.path.join('./metadata.csv', os.path.basename(image_path))  # Create new image path
        os.makedirs('./metadata.csv', exist_ok=True)  # Create directory if it doesn't exist
        Image.open(image_path).save(new_image_path)  # Save image to new path

        with open('./metadata.csv', 'a') as f:
            f.write(f"{os.path.basename(image_path).split('.')[0]},{label}\n")  # Append new data to metadata file

        # Retrain the model with the new data
        train_images, val_images, train_labels, val_labels = self.prepare_data('./metadata.csv', './HAM10000_images_part_1')  # Prepare new data
        self.train(
            train_data={'image_paths': train_images, 'labels': train_labels},  # Training data
            val_data={'image_paths': val_images, 'labels': val_labels},  # Validation data
            epochs=10,  # Number of epochs
            batch_size=32,  # Batch size
            learning_rate=0.001  # Learning rate
        )  # Retrain the model