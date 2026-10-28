from aiohttp import web_fileresponse
from typing import List, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

# 1. Define a rich, nested Pydantic Schema with Field descriptions
class SecurityVulnerability(BaseModel):
    cve_or_topic: str = Field(description="Identifier or name of the vulnerability/issue")
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Assessed Risk Level")
    affected_component: str = Field(description="The Library, service, or code pattern effected")
    remediation_steps: List[str] = Field(description="Actionable steps to fix the issue")


class AuditReport(BaseModel):
    summary: str = Field(description="Executive summary of the audit")
    overall_health_score: int = Field(description="Score from 1 (worst) to 100 (best)")
    vulnerabilities: List[SecurityVulnerability]


# 2. Generate Structured Output (Using response.parsed for automatic Pydantic instantiation)
def run_code_audit(code_snippet: str) -> AuditReport:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=f"Perform a security audit on the following Python Code:\n\n{code_snippet}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            responseSchema=AuditReport,
            temperature=0.1
        ),

    )
     # The SDK automatically validates and parses the JSON into your Pydantic class!
    report: AuditReport = response.parsed
    return report

# 3. Combining Structured Outputs with Google Search Grounding (Gemini 3 & 4 feature)
class CompetitorBrief(BaseModel):
    company_name: str
    latest_product_launch: str = Field(description="Most Recent Major Release Found Via Search")
    key_differentiators: List[str]


def research_company_structured(company: str) -> CompetitorBrief:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=f"Search for the latest announcements and product launches from {company}",
        config={
            "tools": [{"google_search": {}}],
            "response_mime_type": "application/json",
            "response_json_schema": CompetitorBrief.model_json_schema(),
        },
    )


    # 2. Extract and print token usage summary
    usage = response.usage_metadata
    print("\n" + "=" * 35)
    print("       TOKEN USAGE SUMMARY")
    print("=" * 35)
    print(f"Input (Prompt) Tokens : {usage.prompt_token_count}")
    print(f"Output Tokens         : {usage.candidates_token_count}")
    print(f"Total Tokens          : {usage.total_token_count}")
    print("=" * 35)

    return CompetitorBrief.model_validate_json(response.text), usage


if __name__ == "__main__":
    # sample_code = """
    # import sqlite3, subprocess
    # def get_user(username, cmd):
    #     conn = sqlite3.connect('users.db')
    #     cursor = conn.cursor()
    #     cursor.execute(f"SELECT * FROM users WHERE name = '{username}'")
    #     subprocess.Popen(cmd, shell=True)
    # """
    # audit = run_code_audit(sample_code)
    # print(f"Health Score: {audit.overall_health_score}/100")
    # for v in audit.vulnerabilities:
    #     print(f" - [{v.severity}] {v.cve_or_topic}: {v.remediation_steps[0]}")


    brief, usage = research_company_structured("apple")

    print(usage, brief) 



    
    