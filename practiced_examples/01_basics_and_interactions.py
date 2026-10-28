from PIL import GimpGradientFile
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os


load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL")

def get_client() -> genai.Client:
    """
    Creates and returns a unified GenAI Client.
    Automatically picks up GEMINI_API_KEY or GOOGLE_API_KEY from the environment.
    """
    return genai.Client(
        http_options=types.HttpOptions(
            headers={"x-goog-api-client": "my-antigravity-app/1.0.0"}
        )
    )

if __name__ == "__main__":
    client = get_client()
    try:
        response = client.models._generate_content(
            model=MODEL,
            contents="Explain the unified client architecture in two sentances"
        )
        print(response.text)
    finally:
            client.close()
    

