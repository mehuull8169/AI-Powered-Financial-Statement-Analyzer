import os
from huggingface_hub import InferenceClient
from typing import List, Dict
import json
import logging
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Configure logging for better error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze the sentiment of financial text using FinBERT.

    Args:
        text (str): Financial text to analyze

    Returns:
        Dict[str, float]: Dictionary with sentiment label and score
    """
    try:
        # 1. Retrieve API Token
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("Hugging Face API token (HF_TOKEN) not found in environment variables.")

        # 2. Initialize Inference Client
        client = InferenceClient(
            "ProsusAI/finbert",
            token=hf_token
        )

        # 3. Call the Inference API
        result = client.post(
            json={"inputs": text},
            task="text-classification"
        )
            
        # 4. Decode bytes response and extract sentiment
        if isinstance(result, bytes):
            decoded = json.loads(result.decode('utf-8'))
            logging.info(f"API Response: {decoded}")  # Log the response for debugging
            
            # Handle different response formats
            if isinstance(decoded, list) and len(decoded) > 0:
                if isinstance(decoded[0], list) and len(decoded[0]) > 0:
                    # Handle nested list format
                    sentiment = decoded[0][0]
                else:
                    # Handle flat list format
                    sentiment = decoded[0]
                    
                return {
                    'label': sentiment['label'].lower(),  # Convert to lowercase
                    'score': float(sentiment['score'])
                }
            else:
                raise ValueError(f"Unexpected response format: {decoded}")
            
        raise ValueError(f"Expected bytes response, got {type(result)}")

    except Exception as e:
        logging.exception("An unexpected error occurred during sentiment analysis:")
        raise Exception(f"Error analyzing sentiment: {str(e)}")


if __name__ == "__main__":
    # Example usage
    sample_text = '''Dear Shareholders,
    In 2024, our company navigated a challenging economic environment with resilience. 
    Despite fluctuations in global markets, we achieved consistent growth across all segments.
    Our new product line contributed significantly to revenue growth, and we expect continued 
    positive momentum in 2025. However, uncertainties around regulatory changes and inflation 
    remain potential risks to our long-term goals.
    '''

    try:
        result = analyze_sentiment(sample_text)
        print(f"Label: {result['label']}")
        print(f"Score: {result['score']:.4f}")
    except Exception as e:
        print(f"Error in main: {e}")