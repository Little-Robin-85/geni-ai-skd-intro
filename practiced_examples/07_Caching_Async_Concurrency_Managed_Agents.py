import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

load_dotenv()
client = genai.Client()

#region Solution A: create_and_query_knowledge_cache
def create_and_query_knowledge_cache(large_file_path: str, questions: list[str]):
    # 1. Upload the heavy document via Files API
    uploaded_doc = client.files.upload(file=large_file_path)

    # 2. Pin the document and system instruction into a Cache for 30 minutes (1800s)
    cache = client.caches.create(
        model="gemini-3.5-flash-lite",
        config=types.CreateCachedContentConfig(
            display_name="arch_manual_cache",
            system_instruction="You are a technical assistant answering strictly from the cached manual.",
            contents=[uploaded_doc],
            ttl="240s",
        ),
    )

    print(f"Cache created: {cache.name} (expires: {cache.expire_time})")

    for q in questions:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=q,
            config=types.GenerateContentConfig(cached_content=cache.name))
            
        print(f"\nQ: {q}\nA: {response.text}")
        print(f"Cached Tokens Reused: {response.usage_metadata.cached_content_token_count}")

    # 2. Extract and print token usage summary
    usage = response.usage_metadata
    print("\n" + "=" * 35)
    print("       TOKEN USAGE SUMMARY")
    print("=" * 35)
    print(f"Input (Prompt) Tokens : {usage.prompt_token_count}")
    print(f"Output Tokens         : {usage.candidates_token_count}")
    print(f"Total Tokens          : {usage.total_token_count}")
    print("=" * 35)

    print(usage)

# create_and_query_knowledge_cache("/home/ljdeveney/Downloads/membership.pdf",
#                                 ["What is the purpose of this document?", "how many memebrs are women in percentage?"])
#endregion 

#region Solution B: analyze_topic_async
async def analyze_topic_async(aclient, topic: str, t0: float, task_id: int) -> str:
    """Runs a single non-blocking call with lifecycle tracing."""
    send_time = time.perf_counter()
    print(f"[{send_time - t0:6.2f}s] [Task {task_id}] 🚀 DISPATCHED: '{topic}'")
    response = await aclient.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=f"Provide a 2-sentence executive summary of: {topic}",
    )
    done_time = time.perf_counter()
    duration = done_time - send_time
    print(f"[{done_time - t0:6.2f}s] [Task {task_id}] ✅ COMPLETED in {duration:.2f}s: '{topic}'")
    return f"[{topic}]: {response.text.strip()}"
async def batch_process_concurrently(topics: list[str]):
    t0 = time.perf_counter()
    print(f"[{0.00:6.2f}s] Starting batch execution of {len(topics)} topics...")
    # Keep asyncio.gather INSIDE the context manager so the session stays open
    async with genai.Client().aio as aclient:
        tasks = [
            analyze_topic_async(aclient, topic, t0, idx + 1) 
            for idx, topic in enumerate(topics)
        ]
        
        # All tasks execute concurrently across the event loop
        results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t0
    print(f"\n{'='*55}")
    print(f"🎉 All {len(topics)} tasks finished in {total_time:.2f}s total wall-clock time!")
    print(f"{'='*55}\n")
    for res in results:
        print(res, "\n")
if __name__ == "__main__":
    topics_to_run = [
        "Zero-Knowledge Proofs in 2026",
        "WebAssembly Component Model",
        "Python Free-Threaded Mode (No-GIL)",
        "Retrieval-Augmented Generation vs Context Caching",
    ]
    asyncio.run(batch_process_concurrently(topics_to_run))
#endregion

