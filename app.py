from flask import Flask, request, render_template  # Import Flask and necessary modules for handling requests and rendering templates
from Model import SkinCancerModel  # Import the SkinCancerModel class from Model.py
from nlp_ollama import LangChainOllama  # Import the LangChainOllama class from nlp_ollama.py
import os  # Import os for file and directory operations
import requests  # Add this import
import csv
from datetime import datetime

# Initialize the Flask application
app = Flask(__name__)  # Create a Flask application instance

# Initialize model and NLP components
skin_cancer_model = SkinCancerModel()  # Create an instance of SkinCancerModel
nlp_model = LangChainOllama()  # Create an instance of LangChainOllama

# New function to save prediction to CSV
def save_prediction_to_csv(image_path, predicted_label, confidence):
    csv_file = 'predictions.csv'
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Image', 'Prediction', 'Confidence'])
        writer.writerow([datetime.now(), image_path, predicted_label, confidence])

# Route for the home page with two options: Ask a question or upload an image
@app.route('/')
def home():
    return render_template('index.html')  # Render the home page template

# Route to handle skin cancer-related questions using Ollama
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question')  # Get the question from the form data
    if question:
        response = nlp_model.generate_response(question)  # Generate a response using the NLP model
        if response.startswith("Error:"):
            return render_template('error.html', error=response), 503  # Render the error page with the error message
        return render_template('response.html', response=response)  # Render the response page with the generated response
    return 'No question provided', 400  # Return an error if no question is provided

# Route to handle skin cancer image upload for detection
@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    try:
        predicted_label, confidence = skin_cancer_model.predict(image_path)

        save_prediction_to_csv(image_path, predicted_label, confidence)
        
        # Check if it's time to retrain (now every 5 predictions)
        with open('predictions.csv', 'r') as f:
            num_predictions = sum(1 for line in f) - 1  # Subtract 1 for header
        
        if num_predictions % 5 == 0:  # Changed from 10 to 5
            skin_cancer_model.retrain_with_new_data('predictions.csv')

        description = nlp_model.generate_response("What does this mean?", predicted_label, confidence)

        if description.startswith("Error:"):
            return render_template('error.html', error=description), 503

        return render_template('result.html', label=predicted_label, confidence=f"{confidence * 100:.2f}%", description=description)
    except requests.exceptions.ConnectionError:
        error_message = "Unable to connect to the NLP server. Please ensure it's running and try again."
        return render_template('error.html', error=error_message), 503

@app.route('/train', methods=['POST'])
def train_model():
    metadata_path = request.form.get('metadata_path')  # Get the metadata path from the form data
    images_dir = request.form.get('images_dir')  # Get the images directory from the form data
    if not metadata_path or not images_dir:
        return 'No metadata path or images directory provided', 400  # Return an error if metadata path or images directory is not provided

    train_images, val_images, train_labels, val_labels = SkinCancerModel.prepare_data(metadata_path, images_dir)  # Prepare the data for training
    SkinCancerModel.train(
        train_data={'image_paths': train_images, 'labels': train_labels},  # Training data
        val_data={'image_paths': val_images, 'labels': val_labels},  # Validation data
        epochs=10,  # Number of epochs
        batch_size=32,  # Batch size
        learning_rate=0.001  # Learning rate
    )  # Train the model
    return 'Model trained successfully', 200  # Return success message

@app.route('/add_data', methods=['POST'])
def add_data():
    if 'file' not in request.files or 'label' not in request.form:
        return 'No file or label provided', 400  # Return an error if no file or label is provided

    file = request.files['file']  # Get the uploaded file
    label = request.form.get('label')  # Get the label from the form data
    if file.filename == '':
        return 'No selected file', 400  # Return an error if no file is selected

    # Save the uploaded file to a folder
    image_path = os.path.join('uploads', file.filename)  # Create the path to save the uploaded file
    file.save(image_path)  # Save the file to the specified path

    # Add new data to the training set and retrain the model
    skin_cancer_model.add_new_data(image_path, label)  # Add new data and retrain the model

    return 'New data added and model retrained successfully', 200  # Return success message

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask application in debug mode
