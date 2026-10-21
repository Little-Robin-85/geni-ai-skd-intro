# Agent Guidelines: Modern Google GenAI SDK Patterns

This repository strictly uses the modern **Google Gen AI SDK** (`google-genai>=2.30.1`).
All agents, sub-agents, and developers working in this workspace must adhere to the following architectural rules and standards.

---

## 1. Mandatory SDK & Import Conventions

- **Package requirement:** `google-genai>=2.30.1` (optionally with `[aiohttp]` extra).
- **Mandatory imports:**
  ```python
  from google import genai
  from google.genai import types
  ```
- **Prohibited legacy packages & patterns:**
  - **NEVER** install or import `google.generativeai` or `google-generativeai`.
  - **NEVER** use `genai.configure(api_key=...)`.
  - **NEVER** use `genai.GenerativeModel(...)` or `model.generate_content(...)`.
  - **ALWAYS** initialize and use the unified client architecture:
    ```python
    client = genai.Client()
    ```

---

## 2. Model Catalog (2026 Standards)

Always use modern 2026 model identifiers:
- `gemini-3.8-flash`: Default workhorse model for general generation, streaming, multimodal tasks, and tool use.
- `gemini-4-argon`: Flagship reasoning model with extended thinking and long context decoding.
- `gemini-3.5-flash-lite`: Ultra-fast, cost-efficient model for high-throughput structured extraction.
- `antigravity-preview-09-2026`: Remote sandbox managed agent with bash, python, and web tooling via the Interactions API.

> **Deprecated models:** Do NOT use `gemini-1.5-*`, `gemini-2.0-*`, or `gemini-2.5-*`. If requested, substitute with `gemini-3.8-flash`.

---

## 3. Two Primary SDK Paradigms

### A. The Interactions API (`client.interactions.create`)
Best for:
- Server-side stateful multi-turn conversations (`previous_interaction_id=...`).
- Managed Agents (such as `antigravity-preview-09-2026`, `deep-research-preview-04-2026`) with `environment="remote"`.
- Turn-by-turn orchestration and structured event timeline inspection.
- Streaming with `stream=True` yielding typed event steps (`step.start`, `step.delta`, `interaction.completed`).

```python
# Single turn
interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Tell me a joke.",
)
print(interaction.output_text)

# Stateful follow-up turn
turn2 = client.interactions.create(
    model="gemini-3.8-flash",
    input="Explain why that is funny.",
    previous_interaction_id=interaction.id,
)
print(turn2.output_text)
```

### B. The Models API (`client.models.generate_content` / `generate_content_stream`)
Best for:
- Fine-grained generation configuration via `types.GenerateContentConfig`.
- Thinking control via `types.ThinkingConfig(thinking_level="low"|"medium"|"high")`.
- Structured outputs with Pydantic (`response_schema=MyModel`, accessed via `response.parsed`).
- Automatic Python function calling (`tools=[func1, func2]`).
- Explicit Context Caching (`client.caches.create()`).
- High-performance asynchronous execution via `async with client.aio as aclient:`.

```python
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Analyze this architecture.",
    config=types.GenerateContentConfig(
        system_instruction="You are a principal cloud architect.",
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    ),
)
print(response.text)
```

---

## 4. Documentation References

- **Gemini Docs Index (Fallback & LLM Reference):**
  `https://ai.google.dev/gemini-api/docs/llms.txt`
- **Interactions API Reference:**
  `https://ai.google.dev/gemini-api/docs/interactions.md.txt`
- **Managed Agents Quickstart:**
  `https://ai.google.dev/gemini-api/docs/managed-agents-quickstart.md.txt`
- **Structured Outputs & Schema:**
  `https://ai.google.dev/gemini-api/docs/structured-output.md.txt`
