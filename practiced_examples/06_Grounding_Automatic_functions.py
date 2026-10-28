from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

#region ask_with_verified_sources

def ask_with_verified_sources(query: str):
    config = types.GenerateContentConfig(
        tools=[
            types.Tool(google_search=types.GoogleSearch()),
            types.Tool(url_context=types.UrlContext()), # Lets Gemini read URL's in Prompt
        ]
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=query,
        config=config
    )

    print("Answer:\n", response.text)

    # Extract search queries and source URLs from Grounding Metadata
    metadata = response.candidates[0].grounding_metadata
    if metadata and metadata.web_search_queries:
        print("\nQueries Executed:", metadata.web_search_queries)
    if metadata and metadata.grounding_chunks:
        print("\nVerifed Sources")
        for chunk in metadata.grounding_chunks:
            if chunk.web:
                print(f" - {chunk.web.title}: {chunk.web.uri}")

    # 2. Extract and print token usage summary
    usage = response.usage_metadata
    print("\n" + "=" * 35)
    print("       TOKEN USAGE SUMMARY")
    print("=" * 35)
    print(f"Input (Prompt) Tokens : {usage.prompt_token_count}")
    print(f"Output Tokens         : {usage.candidates_token_count}")
    print(f"Total Tokens          : {usage.total_token_count}")
    print("=" * 35)

#ask_with_verified_sources("Who was William Gladstone?")

#endregion

def get_server_metrics(cluster_name: str) -> dict:
    """
    Fetches real-time CPU and memory utilization for a given server cluster.

    Args:
        cluster_name: The identifier of the cluster (e.g., 'eu-west-prod', 'us-east-staging').

    Returns:
        A dictionary containing cpu_percent, memory_percent, and status.
    """
    print(f"[Local Tool Executed] -> get_server_metrics(cluster_name='{cluster_name}')")
    # Mock telemetry lookup
    mock_db = {
        "eu-west-prod": {"cpu_percent": 94.2, "memory_percent": 88.5, "status": "degraded"},
        "us-east-staging": {"cpu_percent": 22.1, "memory_percent": 41.0, "status": "healthy"},
    }
    return mock_db.get(cluster_name, {"error": "Cluster not found"})

def scale_clyster_replicas(cluster_name: str, target_replicas: int) -> dict:
    """
    Scales the number of running container replicas in a cluster.

    Args:
        cluster_name: The identifier of the cluster to scale.
        target_replicas: Desired integer number of replicas (between 1 and 20).

    Returns:
        A confirmation dictionary with the new replica count.
    """
    print(f"[Local Tool Executed] -> scale_cluster_replicas(cluster_name='{cluster_name}', target_replicas={target_replicas})")
    return {"cluster": cluster_name, "new_replicas": target_replicas, "result": "scaled_successfully"}


def run_devops_assistant():
    # Pass the actual Python functions directly into GenerateContentConfig.tools
    config = types.GenerateContentConfig(
        tools=[get_server_metrics, scale_clyster_replicas],
        temperature=0.0,
    )

    prompt = ("Check the metrics for both 'eu-west-prod' and 'us-east-staging'. "
        "If any cluster has CPU usage over 85%, scale its replicas to 10.")

    response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=config,
        )

    print("\nFinal Assistant Response:\n", response.text)

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

if __name__ == "__main__":
    run_devops_assistant()
