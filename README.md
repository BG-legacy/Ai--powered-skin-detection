# Skin Cancer Detection System

## Overview
The Skin Cancer Detection System is a web application designed to assist in the detection and diagnosis of skin cancer using deep learning models. It allows users to upload images of skin lesions and receive predictions along with confidence scores and explanations.

## Features
- Upload images for skin cancer detection.
- Receive predictions with confidence scores.
- Get detailed explanations for each diagnosis.
- Train the model using the HAM10000 dataset.
- Add new data to retrain the model.

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- CUDA (optional, for GPU acceleration)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/skin-cancer-detection.git
   cd skin-cancer-detection
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation
1. **Download the HAM10000 dataset:**
   - Place the metadata CSV file (`HAM10000_metadata.csv`) and image directories (`HAM10000_images_part_1`, `HAM10000_images_part_2`) in the project root.

2. **Verify dataset paths:**
   - Ensure the paths in `train_model.py` and `app.py` are correct for your setup.

## Training the Model
1. **Run the training script:**
   ```bash
   python train_model.py
   ```
   - This will train the model using the HAM10000 dataset and save the trained model weights.

## Running the Web Application
1. **Start the Flask application:**
   ```bash
   python app.py
   ```
   - The application will be accessible at `http://localhost:5001`.

## Usage
- **Upload an Image:**
  - Navigate to the home page and upload an image for analysis.
  - View the prediction result with confidence and explanation.

- **Ask a Question:**
  - Use the question form to get explanations related to skin cancer.

- **Train the Model:**
  - Use the `/train` endpoint to retrain the model with new data.

- **Add New Data:**
  - Use the `/add_data` endpoint to add new images and labels to the dataset.

## Contributing
- Contributions are welcome! Please fork the repository and submit a pull request.

## License
- This project is licensed under the MIT License.

## Contact
- For any questions or issues, please contact [your-email@example.com].
