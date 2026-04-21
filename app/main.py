import json

import requests
from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

# ------------------------
# Tool
# ------------------------

@tool
def get_weather(location: str) -> str:
    """Get current weather for a given city."""
    res = requests.get(f"http://wttr.in/{location}?format=j1")
    data = res.json()["current_condition"][0]

    return f"""
Temperature: {data['temp_C']}°C
Feels like: {data['FeelsLikeC']}°C
Weather: {data['weatherDesc'][0]['value']}
Humidity: {data['humidity']}%
"""

# ------------------------
# LLM
# ------------------------

llm = ChatOllama(
    model="gpt-oss:20b-cloud",
    temperature=1,
)

# ------------------------
# State
# ------------------------

class AgentState(TypedDict):
    input: str
    location: str
    weather: str
    output: str

# ------------------------
# Nodes
# ------------------------

def router(state: AgentState):
    """Decide if weather tool is needed."""
    user_input = state["input"].lower()

    if "weather" in user_input:
        # naive extraction (you can improve later)
        return {"location": "Jakarta"}
    
    return {"output": "I can help with weather questions only."}

def extract_location(state: AgentState):
    """Extract location from user input."""
    response = llm.invoke(f"""
Extract the city name from this sentence.

Sentence: {state['input']}

Only return the city name. No explanation.
""")
    city = response.content.strip()
    if city:
        return {"location": city}
    else: return {"location": "Jakarta"}  # default location


def call_tool(state: AgentState):
    """Call weather tool."""
    weather = get_weather.invoke({"location": state["location"]})
    return {"weather": weather}


def generate_answer(state: AgentState):
    """Generate final answer using LLM."""
    response = llm.invoke(f"""
User question:
{state['input']}

Weather data:
{state['weather']}

Give a decision wether it is suitable for running outdoor activities or not based on the weather data.

You must return a JSON object with keys 'input', 'location', 'decision', and 'output'.
Rules: decision should be either 'suitable' or 'not suitable' based on the weather data. If the temperature is above 30°C, or if there is rain, then it is 'not suitable'. Otherwise, it is 'suitable'.
no extra text outside the JSON object.

The input must be the same as the user question, location must be the city name, and output must be a brief explanation of the decision.
""")
    
    try:
        data = json.loads(response.content)
    except:
        data = {
            "input": state["input"],
            "location": state["location"],
            "decision": "suitable",
            "output": response.content
        }
    return {"output": data}

# ------------------------
# Graph
# ------------------------

graph = StateGraph(AgentState)

graph.add_node("router", router)
graph.add_node("extract_location", extract_location)
graph.add_node("tool", call_tool)
graph.add_node("llm", generate_answer)

# flow
graph.set_entry_point("router")
graph.add_edge("router", "extract_location")
graph.add_edge("extract_location", "tool")
graph.add_edge("tool", "llm")
graph.add_edge("llm", END)

app = graph.compile()

# ------------------------
# Run
# ------------------------

class AnswerWithTemplate(BaseModel):
    input: str
    location: str
    decision: str
    output: str

result = app.invoke({
    "input": "What is the current weather in Bandung?"
})

print(result["output"])