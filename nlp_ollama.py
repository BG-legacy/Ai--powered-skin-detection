# Import necessary modules for LangChain
from langchain_community.chat_models import ChatOllama  # Import ChatOllama model from langchain_community
from langchain_core.output_parsers import StrOutputParser  # Import string output parser from langchain_core
from langchain_core.prompts import ChatPromptTemplate  # Import chat prompt template from langchain_core
import requests

class LangChainOllama:
    def __init__(self):
        try:
            # Initialize the ChatOllama model
            self.llm = ChatOllama(model='llama3.1')  # Create an instance of ChatOllama with the specified model version
        except requests.exceptions.ConnectionError:
            print("Warning: Unable to connect to Ollama server. Responses will be unavailable.")
            self.llm = None

    def generate_response(self, question, prediction=None, confidence=None):
        if self.llm is None:
            return "Error: Ollama server is not available. Please ensure it's running and try again."

        # Format confidence as a percentage if provided
        confidence_percentage = f"{confidence * 100:.2f}%" if confidence is not None else "N/A"

        # Create a ChatPromptTemplate to handle skin cancer queries
        prompt = ChatPromptTemplate.from_template(f"""
        You are a dermatology expert. Provide a detailed and informative response to the following question about skin cancer:
        
        Question: {question}
        
        Model Prediction: {prediction}
        Confidence: {confidence_percentage}
        
        General Description: {prediction} is a type of cancer that develops from the pigment-containing cells known as melanocytes. It is less common than other skin cancers but more dangerous if not detected early.
        
        Include:
        1. General description of {prediction}
        2. Detailed description
        3. Key characteristics
        4. Potential risks
        5. Advice for patients
        6. When to seek medical attention
        
        Be thorough and specific in your response.
        """)  # Define a detailed prompt template for generating a response about skin cancer

        # Create a processing chain (connect the prompt to the LLM and output parser)
        chain = prompt | self.llm | StrOutputParser()  # Chain the prompt, LLM, and output parser together

        try:
            # Generate the response
            response = chain.invoke({'question': question, 'prediction': prediction, 'confidence_percentage': confidence_percentage})
            return response  # Return the generated response
        except requests.exceptions.ConnectionError:
            return "Error: Lost connection to Ollama server. Please try again later."
