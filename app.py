from flask import Flask, request, render_template  # Import Flask and necessary modules for handling requests and rendering templates
from Model import SkinCancerModel  # Import the SkinCancerModel class from Model.py
from nlp_ollama import SimpleDiagnosisExplainer  # Import the SimpleDiagnosisExplainer class from nlp_ollama.py
import os  # Import os for file and directory operations
import requests  # Add this import

# Initialize the Flask application
app = Flask(__name__)  # Create a Flask application instance

# Initialize model and NLP components
skin_cancer_model = SkinCancerModel()  # Create an instance of SkinCancerModel
diagnosis_explainer = SimpleDiagnosisExplainer()  # Create an instance of SimpleDiagnosisExplainer

# Route for the home page with two options: Ask a question or upload an image
@app.route('/')
def home():
    return render_template('index.html')  # Render the home page template

# Route to handle skin cancer-related questions using Ollama
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question')  # Get the question from the form data
    if question:
        response = diagnosis_explainer.generate_response(question)  # Generate a response using the NLP model
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

        if predicted_label == "Please upload an image containing human skin.":
            return render_template('error.html', error=predicted_label), 400

        description = diagnosis_explainer.generate_response("What does this mean?", predicted_label)

        if description.startswith("Error:"):
            return render_template('error.html', error=description), 503

        return render_template('result.html', label=predicted_label, description=description, confidence=confidence)
    except requests.exceptions.ConnectionError:
        error_message = "Unable to connect to the NLP server. Please ensure it's running and try again."
        return render_template('error.html', error=error_message), 503

@app.route('/train', methods=['POST'])
def train_model():
    metadata_path = request.form.get('metadata_path')
    images_dir = request.form.get('images_dir')
    if not metadata_path or not images_dir:
        return 'No metadata path or images directory provided', 400

    model = SkinCancerModel()
    model.train_with_ham10000(metadata_path, images_dir)
    return 'Model trained successfully', 200

@app.route('/add_data', methods=['POST'])
def add_data():
    if 'file' not in request.files or 'label' not in request.form:
        return 'No file or label provided', 400

    file = request.files['file']
    label = request.form.get('label')
    if file.filename == '':
        return 'No selected file', 400

    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    skin_cancer_model.add_new_data(image_path, label)

    return 'New data added and model retrained successfully', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Changed port to 5001
