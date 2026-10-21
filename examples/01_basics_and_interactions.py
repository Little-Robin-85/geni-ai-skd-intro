"""01_basics_and_interactions.py

Demonstrates:
1. Streaming text generation using client.models.generate_content_stream() with
   system instructions and low-budget thinking configuration.
2. Server-side stateful multi-turn conversation using client.interactions.create()
   with previous_interaction_id chaining.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when executed directly from subdirectories
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.genai import types

from src.config import FLASH_MODEL, get_client


def demonstrate_streaming_generation() -> None:
    """Stream text generation using client.models.generate_content_stream()."""
    print("\n--- Paradigm 1: Streaming Content Generation (Models API) ---")
    client = get_client()

    config = types.GenerateContentConfig(
        system_instruction="You are a concise and pragmatic technical assistant.",
        thinking_config=types.ThinkingConfig(thinking_level="low"),
        max_output_tokens=256,
        temperature=0.7,
    )

    prompt = "Explain in two sentences why decoupling compute and state matters in AI agents."
    print(f"Prompt: {prompt}\nStreaming response: ", end="", flush=True)

    response_stream = client.models.generate_content_stream(
        model=FLASH_MODEL,
        contents=prompt,
        config=config,
    )

    for chunk in response_stream:
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n")


def demonstrate_stateful_interactions() -> None:
    """Multi-turn stateful conversation using client.interactions.create()."""
    print("\n--- Paradigm 2: Server-Side Stateful Conversation (Interactions API) ---")
    client = get_client()

    # Turn 1
    print("Turn 1: Introducing user context...")
    turn1 = client.interactions.create(
        model=FLASH_MODEL,
        input="Hello! My name is Liam and I am setting up modern Gemini SDK agent workflows in Antigravity.",
    )
    print(f"Interaction 1 ID: {turn1.id}")
    print(f"Model Turn 1: {turn1.output_text}\n")

    # Turn 2: Referencing previous_interaction_id ensures server maintains memory
    print("Turn 2: Querying server-retained memory...")
    turn2 = client.interactions.create(
        model=FLASH_MODEL,
        input="What is my name, and what specific IDE environment am I setting up?",
        previous_interaction_id=turn1.id,
    )
    print(f"Interaction 2 ID: {turn2.id}")
    print(f"Model Turn 2: {turn2.output_text}\n")


def main() -> None:
    print("==================================================")
    print("Running Example 01: Basics & Interactions")
    print("==================================================")
    demonstrate_streaming_generation()
    demonstrate_stateful_interactions()


if __name__ == "__main__":
    main()
