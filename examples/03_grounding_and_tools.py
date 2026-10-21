"""03_grounding_and_tools.py

Demonstrates:
1. Google Search Grounding with types.Tool(google_search=types.GoogleSearch()) and
   extracting citation metadata from response.candidates[0].grounding_metadata.
2. Python Automatic Function Calling with typed Python functions having Google-style docstrings,
   passed directly to types.GenerateContentConfig(tools=[fn1, fn2]).
"""

import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when executed directly from subdirectories
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.genai import types

from src.config import FLASH_MODEL, get_client


# ---------------------------------------------------------------------------
# 1. Google Search Grounding
# ---------------------------------------------------------------------------
def demonstrate_search_grounding() -> None:
    """Demonstrate real-time Google Search grounding with citation extraction."""
    print("\n--- 1. Grounding with Google Search ---")
    client = get_client()

    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.3,
    )

    query = "What are the latest official Google Gemini models and major AI announcements in 2026?"
    print(f"Submitting grounded search query: '{query}'...")

    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=query,
        config=config,
    )

    print("\nModel Response:")
    print(response.text)

    # Extract citations and search queries from grounding metadata
    if response.candidates and response.candidates[0].grounding_metadata:
        metadata = response.candidates[0].grounding_metadata
        print("\n--- Grounding Metadata & Citations ---")

        if metadata.web_search_queries:
            print("Web Search Queries Executed:")
            for q in metadata.web_search_queries:
                print(f"  • {q}")

        if metadata.grounding_chunks:
            print("\nSource Citations:")
            for idx, chunk in enumerate(metadata.grounding_chunks, 1):
                if chunk.web:
                    title = chunk.web.title or "Untitled Source"
                    uri = chunk.web.uri or "No URI"
                    print(f"  [{idx}] {title}\n      URL: {uri}")


# ---------------------------------------------------------------------------
# 2. Python Automatic Function Calling
# ---------------------------------------------------------------------------
def get_service_telemetry(service_name: str, region: str = "us-central1") -> dict[str, Any]:
    """Retrieves real-time telemetry and p99 latency metrics for an infrastructure microservice.

    Args:
        service_name: The identifier of the microservice (e.g., 'auth', 'payment', 'inference').
        region: The Google Cloud deployment region. Defaults to 'us-central1'.

    Returns:
        A dictionary containing latency_ms, region, error_rate, and operational health status.
    """
    mock_data: dict[str, dict[str, Any]] = {
        "auth": {"p99_latency_ms": 14.2, "error_rate": 0.001, "status": "healthy"},
        "payment": {"p99_latency_ms": 45.8, "error_rate": 0.003, "status": "healthy"},
        "inference": {"p99_latency_ms": 112.5, "error_rate": 0.042, "status": "degraded"},
    }
    data = mock_data.get(
        service_name.lower(),
        {"p99_latency_ms": 25.0, "error_rate": 0.0, "status": "healthy"},
    )
    return {
        "service": service_name,
        "region": region,
        **data,
    }


def trigger_service_remediation(service_name: str, action: str) -> dict[str, str]:
    """Executes a recovery or scaling remediation action for a target microservice.

    Args:
        service_name: The microservice requiring remediation.
        action: The remediation action to execute ('scale_up', 'restart_pods', 'drain_traffic').

    Returns:
        A dictionary confirming execution status and transaction tracking ID.
    """
    return {
        "service": service_name,
        "action": action,
        "status": "initiated",
        "tx_id": f"rem-tx-9942-{service_name}",
        "message": f"Successfully initiated '{action}' on service '{service_name}'.",
    }


def demonstrate_automatic_function_calling() -> None:
    """Demonstrate passing typed Python functions directly into tools for auto-orchestration."""
    print("\n--- 2. Python Automatic Function Calling ---")
    client = get_client()

    # Pass Python functions directly to tools
    config = types.GenerateContentConfig(
        tools=[get_service_telemetry, trigger_service_remediation],
        system_instruction=(
            "You are an SRE incident agent. When asked about system health, inspect the telemetry first. "
            "If the service status is degraded, trigger appropriate remediation ('restart_pods' or 'scale_up') "
            "and explain your resolution."
        ),
        temperature=0.1,
    )

    prompt = (
        "Check the telemetry for the 'inference' service in 'us-central1'. "
        "If it is degraded, initiate remediation."
    )
    print(f"Prompt: '{prompt}'...")

    # The SDK automatically calls the local functions and responds with the completed turn
    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=prompt,
        config=config,
    )

    print("\nFinal Model Resolution (after automatic function execution):")
    print(response.text)


def main() -> None:
    print("==================================================")
    print("Running Example 03: Grounding & Automatic Tools")
    print("==================================================")
    demonstrate_search_grounding()
    demonstrate_automatic_function_calling()


if __name__ == "__main__":
    main()
