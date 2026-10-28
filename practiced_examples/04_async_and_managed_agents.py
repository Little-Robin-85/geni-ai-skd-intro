from aiohttp import web_fileresponse
from http.client import responses
from aiohttp import multipart
from aiohappyeyeballs import utils
from PIL import GimpGradientFile
import time
import httpx
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

# 1. Local Image via Pillow (Automatically converted by the SDK)
def analyse_local_image(image_path: str):
    img = Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=["Extract all visible text and describe the layout of this UI screenshot.", img]
    )

    return response.text

# 2. Remote PDF via Inline Bytes (No local disk write needed!)
def summarize_remote_pdf(pdf_url: str):
    pdf_bytes = httpx.get(pdf_url).content


   
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            "Summarise the key methodology and conclusions in 5 bullet points"
            ]
        )

    # Extract and print token usage summary
    usage = response.usage_metadata
    print("\n" + "=" * 35)
    print("       TOKEN USAGE SUMMARY")
    print("=" * 35)
    print(f"Input (Prompt) Tokens : {usage.prompt_token_count}")
    print(f"Output Tokens         : {usage.candidates_token_count}")
    print(f"Total Tokens          : {usage.total_token_count}")
    print("=" * 35)

    return (response.text, usage)

# 3. Audio/Video Upload via the Files API (With State Polling)
def transcribe_or_analyse_media(file_path: str, prompt: str):
    print(f"Uploading {file_path} to Files API...")
    uploaded_file = client.files.upload(file=file_path)
    uploaded_file = client.files.get(name=uploaded_file.name)

    # Videos and large media require server-side processing before inference
    while uploaded_file.state.name == 'PROCESSING':
        print("Waiting for media processing...")
        time.sleep(2.5)
        uploaded_file = client.files.get(name=uploaded_file.name)

    if uploaded_file.state.name != "ACTIVE":
        raise RuntimeError(f"File processing failed: {uploaded_file.state.name}")

    else:
        # Use gemini-3.1-flash-lite for fast, cheap audio transcription, or 3.8-flash for video
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[prompt, uploaded_file]
        )

        usage = response.usage_metadata
        print("\n" + "=" * 35)
        print("       TOKEN USAGE SUMMARY")
        print("=" * 35)
        print(f"Input (Prompt) Tokens : {usage.prompt_token_count}")
        print(f"Output Tokens         : {usage.candidates_token_count}")
        print(f"Total Tokens          : {usage.total_token_count}")
        print("=" * 35)

        #clean up remote storage when finished
        client.files.delete(name=uploaded_file.name)
        return response.text, usage

answer, tokens = transcribe_or_analyse_media("practiced_examples/Bill_Dotson.mp3",
                            "Transcibe this audio for me and print out the results")

print(answer)
