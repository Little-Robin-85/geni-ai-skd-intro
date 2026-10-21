from dotenv import load_dotenv
from google import genai
from google.genai import types
import os


load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL")

client = genai.Client(api_key=API_KEY)

