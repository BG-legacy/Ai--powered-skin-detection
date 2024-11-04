class SimpleDiagnosisExplainer:
    def __init__(self):
        # Dictionary containing explanations for different skin conditions
        self.explanations = {
            "Actinic Keratoses and Bowen's disease": """
                A pre-cancerous skin growth typically caused by sun damage.
                Key characteristics:
                - Rough, scaly patches on sun-exposed areas
                - May be red, tan, pink, or flesh-colored
                Advice: Protect skin from sun exposure and seek regular dermatological check-ups.
            """,
            "Basal Cell Carcinoma": """
                The most common type of skin cancer.
                Key characteristics:
                - Pearly, waxy bumps
                - Flat, flesh-colored or brown scar-like lesions
                Advice: Early treatment has high success rates. Regular skin checks recommended.
            """,
            "Benign Keratosis-like Lesions": """
                Non-cancerous skin growths that commonly appear with age.
                Key characteristics:
                - Brown, black or light tan
                - Waxy, scaly, slightly raised
                Advice: Generally harmless but monitor for changes.
            """,
            "Dermatofibroma": """
                A common benign skin tumor.
                Key characteristics:
                - Small, firm bump
                - Usually brown to reddish
                Advice: Usually harmless and doesn't require treatment unless symptomatic.
            """,
            "Melanoma": """
                A serious form of skin cancer that develops from melanocytes.
                Key characteristics:
                - Asymmetrical shape
                - Irregular borders
                - Variable colors
                Advice: Requires immediate medical attention. Early detection is crucial.
            """,
            "Melanocytic Nevi": """
                Common moles, usually benign.
                Key characteristics:
                - Round or oval shape
                - Even coloring
                - Clear borders
                Advice: Monitor for changes using the ABCDE rule.
            """,
            "Vascular Lesions": """
                Abnormalities of blood vessels in the skin.
                Key characteristics:
                - Red, purple, or pink in color
                - May be flat or raised
                Advice: Most are harmless but consult doctor if concerned.
            """
        }

    def generate_response(self, question, prediction=None):
        if prediction is None:
            return "Error: No diagnosis provided."
        
        explanation = self.explanations.get(prediction)
        if explanation is None:
            return "Error: Unknown diagnosis type."
        
        return explanation.strip()
