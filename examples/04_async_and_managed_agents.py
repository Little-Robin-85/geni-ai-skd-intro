"""04_async_and_managed_agents.py

Demonstrates:
1. High-concurrency async content generation using async with get_client().aio as aclient:
   and asyncio.gather() across multiple parallel prompts.
2. Invoking the Antigravity Managed Agent (agent="antigravity-preview-09-2026", environment="remote")
   using client.interactions.create() to orchestrate tasks in a sandboxed Linux environment.
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path when executed directly from subdirectories
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.genai import types

from src.config import ANTIGRAVITY_AGENT, FLASH_MODEL, get_client


# ---------------------------------------------------------------------------
# 1. High-Concurrency Asynchronous Generation
# ---------------------------------------------------------------------------
async def demonstrate_concurrent_async_generation() -> None:
    """Run multiple prompt generations concurrently using the async client."""
    print("\n--- 1. High-Concurrency Async Generation (client.aio + asyncio.gather) ---")

    prompts = [
        "Explain the benefit of asynchronous event loops in Python in one punchy sentence.",
        "Give one concrete best practice for rate-limit backoff in distributed LLM architectures.",
        "Summarize how server-side state saves tokens in multi-turn conversations in one sentence.",
    ]

    print(f"Dispatching {len(prompts)} parallel async requests...")

    # Using the async client context manager automatically manages connection pooling
    async with get_client().aio as aclient:
        async def fetch_response(idx: int, prompt: str) -> tuple[int, str]:
            response = await aclient.models.generate_content(
                model=FLASH_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=150,
                    temperature=0.4,
                ),
            )
            return idx, response.text or ""

        tasks = [fetch_response(i + 1, p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

    for idx, text in sorted(results, key=lambda x: x[0]):
        print(f"\n[Response {idx}] Prompt: '{prompts[idx - 1]}'\nOutput: {text.strip()}")


# ---------------------------------------------------------------------------
# 2. Antigravity Managed Agent Invocation
# ---------------------------------------------------------------------------
def invoke_antigravity_agent(
    task_prompt: str = (
        "Write a Python script that calculates the SHA-256 hash of 'Google Gemini 2026', "
        "save it to /workspace/hash.txt, and display the file contents."
    ),
) -> None:
    """Invoke the Antigravity Managed Agent with remote sandbox environment provisioning."""
    print("\n--- 2. Antigravity Managed Agent (Remote Sandbox) ---")
    client = get_client()

    print(f"Deploying task to agent '{ANTIGRAVITY_AGENT}' with environment='remote'...")
    print(f"Task: {task_prompt}")

    interaction = client.interactions.create(
        agent=ANTIGRAVITY_AGENT,
        input=task_prompt,
        environment="remote",
    )

    env_id = getattr(interaction, "environment_id", "remote-sandbox")
    print(f"\nSandbox Environment ID: {env_id}")
    print(f"Interaction ID: {interaction.id}")
    print(f"Status: {interaction.status}")
    print("\nAgent Output:")
    print(interaction.output_text)


def main() -> None:
    print("==================================================")
    print("Running Example 04: Async & Managed Agents")
    print("==================================================")
    asyncio.run(demonstrate_concurrent_async_generation())

    # Note: Antigravity agent execution can be triggered by calling invoke_antigravity_agent()
    # Below is a safe demonstration trigger.
    print("\n[Optional] You can run `invoke_antigravity_agent()` to provision a remote sandbox.")


if __name__ == "__main__":
    main()
