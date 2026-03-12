import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Any
from openai import OpenAI

# Import your friend's updated modules
from med_bot.config import LLMOD_AI_API_KEY, LLMOD_AI_BASE_URL, PC_ABSTRACTS_INDEX_NAME, PC_SYMPTOMS_INDEX_NAME, WEATHER_API_KEY
from med_bot.user_data_db import load_table, sql_command_table
from med_bot.vector_db.embedding import get_embeddings
from med_bot.vector_db.index import get_index, retrieve_matches

app = FastAPI(title="MedBot API")

# Setup LLMod Client
llm_client = OpenAI(api_key=LLMOD_AI_API_KEY, base_url=LLMOD_AI_BASE_URL)


# 1. Pydantic Models for strict JSON validation
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


# 2. MedBot Tools
def tool_pattern_engine(limit: int = 14) -> str:
    """Fetches user data from Supabase to find patterns."""
    try:
        table = load_table()
        rows = sql_command_table(table, "*")
        recent_data = rows[-limit:] if len(rows) > limit else rows
        return f"User Data Logs: {json.dumps(recent_data)}"
    except Exception as e:
        return f"Error retrieving user data: {str(e)}"


def tool_scientific_validator(query: str) -> str:
    """RAG against NIH/PMC abstracts in Pinecone."""
    try:
        embeddings = get_embeddings()
        index = get_index(PC_ABSTRACTS_INDEX_NAME)
        matches = retrieve_matches(index, embeddings, query, top_k=3)
        abstracts = [match["metadata"].get("chunk_text", "") for match in matches]
        return "Scientific Context: " + " | ".join(abstracts)
    except Exception as e:
        return f"Error querying science database: {str(e)}"


def tool_intervention_advisor(condition: str) -> str:
    """RAG against the symptoms/treatments index in Pinecone."""
    try:
        embeddings = get_embeddings()
        index = get_index(PC_SYMPTOMS_INDEX_NAME)
        matches = retrieve_matches(index, embeddings, condition, top_k=3)
        # Safely dumping metadata as JSON since structure might vary
        treatments = [json.dumps(match["metadata"]) for match in matches]
        return "Intervention Context: " + " | ".join(treatments)
    except Exception as e:
        return f"Error querying intervention database: {str(e)}"


def tool_env_scanner(city: str) -> str:
    """Live API call to OpenWeatherMap for environmental data."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            return f"Env Data for {data['name']}: Temp {data['main']['temp']}C, Weather: {data['weather'][0]['description']}"
        else:
            return f"Error fetching weather: {data}"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


# 3. Agent ReAct Logic
def execute_react_agent(user_prompt: str) -> ExecuteResponse:
    steps_trace = []
    max_iterations = 5

    # Updated System Prompt to force autonomous tool usage
    system_prompt = f"""
        You are MedBot, an autonomous causal health investigator.
        Your goal is to transition from mere monitoring to dynamic, inquiry-driven analysis, actively seeking root causes.

        You have access to the following tools:
        1. PatternEngine: Use to fetch the user's personal synthetic health logs. Input: UserID (Always use "user_123" for the current user).
        2. ScientificValidator: Use to search medical literature (NIH/PMC abstracts). Input: search query.
        3. EnvScanner: Use to get live weather/environmental data. Input: city name (e.g., "Haifa" or "Tel Aviv").
        4. InterventionAdvisor: Use to get treatment protocols. Input: condition string.

        CRITICAL INSTRUCTIONS:
        - DO NOT ask the user for permission to review logs. 
        - DO NOT ask for a User ID. Assume the user is "user_123".
        - You MUST autonomously use the PatternEngine, EnvScanner, and ScientificValidator tools to investigate the root cause BEFORE giving a FinalAnswer.
        - Connect the user's personal physiological signals, real-time environmental conditions, and evidence-backed medical knowledge to form your diagnosis.
        - STRICT STEP LIMIT: You only have a maximum of {max_iterations} tool calls allowed. You must efficiently gather your data and output a 'FinalAnswer' before you run out of turns.
        - DO NOT use the exact same tool with the exact same input more than once. If you already fetched the user logs, do not fetch them again.

        You must output a strictly valid JSON object on every turn with the following keys:
        {{
          "thought": "Your reasoning here",
          "tool_name": "Name of tool to use, or 'FinalAnswer' if done",
          "tool_input": "Input for the tool",
          "final_answer": "Your final recommendation to the user (only populate if tool_name is FinalAnswer, otherwise leave null)"
        }}
        """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # max_iterations = 5
    for i in range(max_iterations):
        # LLM Call - Fixed model name to match your specific key access
        response = llm_client.chat.completions.create(
            model="RPRTHPB-gpt-5-mini",
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
            tool_result = tool_pattern_engine(14)
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

    # return ExecuteResponse(status="error", error="Agent exceeded maximum iterations.", steps=steps_trace)
    # --- NEW FALLBACK MECHANISM ---
    # If the loop finishes and we haven't returned a FinalAnswer yet:
    messages.append({
        "role": "user",
        "content": "CRITICAL: You have run out of time and tool calls. You MUST immediately provide a final recommendation based ONLY on the observations gathered so far. Set tool_name to 'FinalAnswer'."
    })

    try:
        fallback_response = llm_client.chat.completions.create(
            model="RPRTHPB-gpt-5-mini",
            messages=messages,
            response_format={"type": "json_object"}
        )
        fallback_output = fallback_response.choices[0].message.content
        fallback_action = json.loads(fallback_output)

        # Extract the final answer, or default to their last thought if they still failed
        final_answer = fallback_action.get("final_answer") or fallback_action.get("thought",
                                                                                  "Could not reach a conclusive diagnosis.")

        steps_trace.append(Step(
            module="Smart LLM Orchestrator (Fallback)",
            prompt="Forced final answer due to iteration limit.",
            response=fallback_output
        ))

        return ExecuteResponse(status="ok", response=f"[Partial Data Warning] {final_answer}", steps=steps_trace)

    except Exception as e:
        return ExecuteResponse(status="error", error=f"Agent exceeded iterations and fallback failed: {str(e)}",
                               steps=steps_trace)


# 4. Required API Endpoints
@app.get("/")
def serve_frontend():
    """Serves the minimal Web UI."""
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse("index.html")


@app.get("/api/team_info", response_model=TeamInfo)
def get_team_info():
    """Returns student details."""
    return TeamInfo(
        group_batch_order_number="batch3_order8",
        team_name="MedBot",
        students=[
            Student(name="Tom Shur", email="tom.shur@campus.technion.ac.il"),
            Student(name="Roei Levi", email="roei.levi@campus.technion.ac.il"),
            Student(name="Or Davidovich", email="ordavidovich@campus.technion.ac.il")
        ]
    )


@app.get("/api/agent_info")
def get_agent_info():
    """Returns agent meta and how to use it."""
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
    """Returns the architecture diagram as an image (PNG)."""
    file_path = "architecture.png"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Architecture diagram not found.")
    return FileResponse(file_path, media_type="image/png")


@app.post("/api/execute", response_model=ExecuteResponse)
def execute_agent(request: ExecuteRequest):
    """Main entry point for the agent."""
    try:
        result = execute_react_agent(request.prompt)
        return result
    except Exception as e:
        return ExecuteResponse(status="error", error=str(e), steps=[])