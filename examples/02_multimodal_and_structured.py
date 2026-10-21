"""02_multimodal_and_structured.py

Demonstrates:
1. Structured Outputs using a nested Pydantic BaseModel schema with Field descriptions,
   response_mime_type="application/json", response_schema, and response.parsed.
2. Inline multimodal image analysis using types.Part.from_bytes() and PIL.Image.
"""

import io
import sys
from pathlib import Path

# Ensure project root is on sys.path when executed directly from subdirectories
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.genai import types
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from src.config import FLASH_MODEL, get_client


# ---------------------------------------------------------------------------
# Structured Output Schemas (Nested Pydantic Models)
# ---------------------------------------------------------------------------
class ServiceComponent(BaseModel):
    name: str = Field(description="Name of the software component or service")
    layer: str = Field(description="Architectural layer (e.g., Ingestion, Reasoning, Storage)")
    technology: str = Field(description="Proposed programming language, library, or GCP service")
    purpose: str = Field(description="Core responsibility of this component")


class ArchitectureReport(BaseModel):
    system_name: str = Field(description="Descriptive name of the proposed system architecture")
    summary: str = Field(description="Executive overview of the system architecture")
    components: list[ServiceComponent] = Field(description="Key decoupled architectural components")
    security_considerations: list[str] = Field(description="Critical security considerations and guardrails")


def demonstrate_structured_output() -> ArchitectureReport:
    """Demonstrate extracting typed, validated Pydantic instances using response_schema."""
    print("\n--- 1. Structured Outputs with Nested Pydantic Models ---")
    client = get_client()

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ArchitectureReport,
        system_instruction="You are a Principal Cloud Solutions Architect.",
        temperature=0.2,
    )

    prompt = (
        "Design a decoupled multi-agent intelligence pipeline on Google Cloud that ingests documents, "
        "indexes embeddings, orchestrates reasoning via Gemini, and exposes a real-time API."
    )
    print(f"Requesting structured architecture plan for: '{prompt}'...")

    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=prompt,
        config=config,
    )

    # response.parsed contains the automatically deserialized Pydantic instance
    report: ArchitectureReport = response.parsed
    print(f"\nExtracted Architecture: {report.system_name}")
    print(f"Summary: {report.summary}\n")
    print("Components:")
    for comp in report.components:
        print(f"  - [{comp.layer}] {comp.name} ({comp.technology}): {comp.purpose}")
    print("\nSecurity Considerations:")
    for sec in report.security_considerations:
        print(f"  - {sec}")

    return report


# ---------------------------------------------------------------------------
# Multimodal Analysis Helpers
# ---------------------------------------------------------------------------
def create_sample_canvas_image() -> bytes:
    """Create a simple in-memory PNG image with geometric patterns for multimodal analysis."""
    img = Image.new("RGB", (320, 240), color=(240, 244, 248))
    draw = ImageDraw.Draw(img)

    # Draw shapes
    draw.rectangle([40, 40, 140, 140], fill=(66, 133, 244), outline=(26, 115, 232), width=3)
    draw.ellipse([180, 40, 280, 140], fill=(52, 168, 83), outline=(30, 142, 62), width=3)
    draw.polygon([(160, 160), (220, 220), (100, 220)], fill=(234, 67, 53), outline=(217, 48, 37), width=3)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def demonstrate_inline_multimodal_analysis() -> None:
    """Demonstrate inline multimodal reasoning using types.Part.from_bytes()."""
    print("\n--- 2. Inline Multimodal Analysis using types.Part.from_bytes ---")
    client = get_client()

    image_bytes = create_sample_canvas_image()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    prompt = (
        "Inspect this image carefully. Identify all geometric shapes present, their colors, "
        "and their spatial arrangement."
    )
    print(f"Sending image ({len(image_bytes)} bytes) + prompt: '{prompt}'...")

    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=300,
        ),
    )

    print("\nModel Analysis:")
    print(response.text)


def main() -> None:
    print("==================================================")
    print("Running Example 02: Multimodal & Structured Output")
    print("==================================================")
    demonstrate_structured_output()
    demonstrate_inline_multimodal_analysis()


if __name__ == "__main__":
    main()
