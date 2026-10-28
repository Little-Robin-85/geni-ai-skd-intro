from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client=genai.Client()

#region genrate_config

#Configure system behaviour ad sampling parameters
config = types.GenerateContentConfig(
    system_instruction=("You are a senior Python architect. Always favor modern type hints, "
        "explain trade-offs concisely, and avoid deprecated libraries."),
        temperature=0.3 # lower temp -> more deterministic
)

print("--- Streaming Response via generate_content_stream ---")
stream = client.models._generate_content_stream(
    model="gemini-3.8-flash",
    contents="What is the difference between" \
     "concurrency and parallelism in Python?",
     config=config
)

for chunk in stream:
    print(chunk.text, end="", flush=True)
print()
#endregion


interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Give me 5 bullet points on why server-side state management simplifies backend code."
)

print(f"Interaction ID: {interaction.id}")
print(f"Output:\n{interaction.output_text}")