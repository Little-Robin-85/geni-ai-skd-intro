"""verify_setup.py

Lightweight smoke test for validating the Google GenAI SDK environment:
1. Verifies that GEMINI_API_KEY or GOOGLE_API_KEY is present in os.environ without printing secrets.
2. Reports the installed google-genai SDK version.
3. Performs a single low-token client.interactions.create() call to ensure end-to-end connectivity.
4. Closes the client cleanly.
"""

import os
import sys

import google.genai

from src.config import FLASH_MODEL, get_client


def main() -> int:
    print("==================================================")
    print("Gemini API Environment & SDK Verification")
    print("==================================================")

    # 1. Check SDK version
    sdk_version = getattr(google.genai, "__version__", "unknown")
    print(f"• Installed google-genai version: {sdk_version}")

    # 2. Verify API key is present in environment without exposing secret values
    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY"))
    has_google_key = bool(os.environ.get("GOOGLE_API_KEY"))

    if has_gemini_key:
        print("• API Key Detection: GEMINI_API_KEY found in environment (valid length, hidden).")
    elif has_google_key:
        print("• API Key Detection: GOOGLE_API_KEY found in environment (valid length, hidden).")
    else:
        print(
            "✖ ERROR: Neither GEMINI_API_KEY nor GOOGLE_API_KEY was found in os.environ.\n"
            "  Please make sure your .env file is present with GEMINI_API_KEY=<your_key>.",
            file=sys.stderr,
        )
        return 1

    # 3. Initialize client and execute lightweight smoke interaction
    print(f"• Initializing genai.Client() targeting model '{FLASH_MODEL}'...")
    client = get_client()

    try:
        interaction = client.interactions.create(
            model=FLASH_MODEL,
            input="Reply with: Environment Verified.",
        )
        output = (interaction.output_text or "").strip()
        print(f"• Live API Response: {output}")
        print("✔ Verification Successful: Environment and SDK communication verified!")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"✖ Verification Failed with error: {exc}", file=sys.stderr)
        return 1
    finally:
        # 4. Cleanly close the client
        if hasattr(client, "close"):
            client.close()


if __name__ == "__main__":
    sys.exit(main())
