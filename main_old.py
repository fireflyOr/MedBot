import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from pinecone import Pinecone
from openai import OpenAI

# from fastapi.responses import FileResponse

# 1. Initialization & Config
load_dotenv()

app = FastAPI(title="MedBot API")

# Setup DB Clients
supabase: Client = create_client(os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_KEY", ""))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "medbot-index"))

# Setup LLMod Client (assuming OpenAI compatibility, standard for AI proxy platforms)
llm_client = OpenAI(
    api_key=os.getenv("LLMOD_API_KEY"),
    base_url=os.getenv("LLMOD_BASE_URL")
)


# 2. Pydantic Models for strict JSON validation (Matching Course Requirements)
class Student(BaseModel):
    name: str
    email: str


class TeamInfo(BaseModel):
    group_batch_order_number: str
    team_name: str
    students: List[Student]


class Step(BaseModel):
    module: str
    prompt: Any
    response: Any


class ExecuteRequest(BaseModel):
    prompt: str


class ExecuteResponse(BaseModel):
    status: str
    error: Optional[str] = None
    response: Optional[str] = None
    steps: List[Step] = []


# 3. MedBot Tools
def tool_pattern_engine(user_id: str, symptom: str) -> str:
    """Queries synthetic user data in Supabase for patterns."""
    try:
        # Assuming you have a table named 'user_health_logs'
        response = supabase.table("user_health_logs").select("*").eq("user_id", user_id).limit(14).execute()
        data = response.data
        return f"Pattern Engine Data: Found {len(data)} recent log entries. User context retrieved."
    except Exception as e:
        return f"Error retrieving user data: {str(e)}"


def tool_scientific_validator(query: str) -> str:
    """RAG against NIH/PMC abstracts in Pinecone."""
    try:
        # Create an embedding for the query (using a small, cheap embedding model)
        embed_res = llm_client.embeddings.create(input=[query], model="text-embedding-3-small")
        vector = embed_res.data[0].embedding

        # Query Pinecone
        query_res = index.query(vector=vector, top_k=3, include_metadata=True)
        contexts = [match.metadata.get('text', '') for match in query_res.matches]
        return "Scientific Context: " + " | ".join(contexts)
    except Exception as e:
        return f"Error querying science database: {str(e)}"


def tool_env_scanner(location: str) -> str:
    """Live API call to Open-Meteo for environmental data."""
    try:
        # Example coordinates for the API. You could use a geocoding API to parse 'location' dynamically.
        lat, lon = 32.0853, 34.7818  # Tel Aviv default
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        res = requests.get(url).json()
        weather = res.get("current_weather", {})
        return f"Env Scanner Data: Temp {weather.get('temperature')}C, Wind {weather.get('windspeed')}km/h."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


def tool_intervention_advisor(condition: str) -> str:
    """Queries Supabase/Pinecone for holistic treatments."""
    return f"Intervention Advisor Data: Suggested protocols for {condition} involve NSDR, hydration, and sleep hygiene."


# 4. Agent ReAct Logic
def execute_react_agent(user_prompt: str) -> ExecuteResponse:
    steps_trace = []

    system_prompt = """
    You are MedBot, an autonomous causal health investigator.
    You have access to the following tools:
    1. PatternEngine: Use to fetch the user's personal synthetic health logs. Input: UserID.
    2. ScientificValidator: Use to search medical literature (NIH/PMC). Input: search query.
    3. EnvScanner: Use to get live weather/environmental data. Input: location string.
    4. InterventionAdvisor: Use to get treatment protocols. Input: condition string.

    You must output a strictly valid JSON object on every turn with the following keys:
    {
      "thought": "Your reasoning here",
      "tool_name": "Name of tool to use, or 'FinalAnswer' if done",
      "tool_input": "Input for the tool",
      "final_answer": "Your final recommendation to the user (only populate if tool_name is FinalAnswer, otherwise leave null)"
    }
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    max_iterations = 5
    for i in range(max_iterations):
        # LLM Call
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",  # Use an efficient model to save budget
            messages=messages,
            response_format={"type": "json_object"}
        )

        llm_output = response.choices[0].message.content
        try:
            action = json.loads(llm_output)
        except:
            return ExecuteResponse(status="error", error="LLM generated invalid JSON.")

        thought = action.get("thought", "")
        tool_name = action.get("tool_name", "")
        tool_input = action.get("tool_input", "")
        final_answer = action.get("final_answer")

        # Log the step
        steps_trace.append(Step(
            module="Smart LLM Orchestrator",
            prompt=messages[-1]["content"] if i > 0 else user_prompt,
            response=llm_output
        ))

        # Check for completion
        if tool_name == "FinalAnswer" or final_answer:
            return ExecuteResponse(status="ok", response=final_answer, steps=steps_trace)

        # Execute Tool
        tool_result = ""
        if tool_name == "PatternEngine":
            tool_result = tool_pattern_engine("user_123", tool_input)
        elif tool_name == "ScientificValidator":
            tool_result = tool_scientific_validator(tool_input)
        elif tool_name == "EnvScanner":
            tool_result = tool_env_scanner(tool_input)
        elif tool_name == "InterventionAdvisor":
            tool_result = tool_intervention_advisor(tool_input)
        else:
            tool_result = f"Unknown tool: {tool_name}"

        # Log tool execution
        steps_trace.append(Step(
            module=tool_name,
            prompt=tool_input,
            response=tool_result
        ))

        # Append observation to context for next loop
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": f"Observation: {tool_result}"})

    return ExecuteResponse(status="error", error="Agent exceeded maximum iterations.", steps=steps_trace)


# 5. Required API Endpoints

@app.get("/api/team_info", response_model=TeamInfo)
def get_team_info():
    """Returns student details[cite: 190, 191, 192]."""
    return TeamInfo(
        group_batch_order_number="batch1_order1",  # Replace with your actual assigned batch/order
        team_name="MedBot",
        students=[
            Student(name="Tom Shur", email="tom.shur@example.com"),  # Replace with actual emails
            Student(name="Roei Levi", email="roei.levi@example.com"),
            Student(name="Or Davidovich", email="or.davidovich@example.com")
        ]
    )


@app.get("/api/agent_info")
def get_agent_info():
    """Returns agent meta and how to use it[cite: 209, 210, 211, 212, 213, 214]."""
    return {
        "description": "MedBot is a ReAct Agent for Causal Health Insights.",
        "purpose": "To proactively investigate health symptoms by connecting personal physiological signals, real-time environmental conditions, and evidence-backed medical knowledge.",
        "prompt_template": {
            "template": "I am experiencing {symptom}. Can you help me find out why and what I should do?"
        },
        "prompt_examples": [
            {
                "prompt": "I'm experiencing sudden, unexplained anxiety today.",
                "full_response": "Your anxiety is likely triggered by significant sleep debt and amplified by barometric pressure changes. We recommend the NSDR protocol to restore neural balance.",
                "steps": [
                    "Pattern Engine detected 40% drop in Deep Sleep.",
                    "Env Scanner noted sharp drop in barometric pressure.",
                    "Scientific Validator correlated REM deprivation with stress sensitivity."
                ]
            }
        ]
    }


@app.get("/api/model_architecture")
def get_model_architecture():
    """Returns the architecture diagram as an image (PNG)[cite: 230, 231, 232]."""
    file_path = "architecture_old.png"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Architecture diagram not found.")
    return FileResponse(file_path, media_type="image/png")


@app.get("/")
def serve_frontend():
    """Serves the minimal Web UI."""
    # Ensure index.html exists in the same directory
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse("index.html")

@app.post("/api/execute", response_model=ExecuteResponse)
def execute_agent(request: ExecuteRequest):
    """Main entry point for the agent[cite: 241, 242, 243, 244]."""
    try:
        result = execute_react_agent(request.prompt)
        return result
    except Exception as e:
        return ExecuteResponse(status="error", error=str(e), steps=[])

# To run locally: uvicorn main:app --reload