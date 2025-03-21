from fastapi import FastAPI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.llms import OpenAI
import openai  # Use OpenAI SDK instead of raw requests

app = FastAPI()

# Research Agent (Web Scraping & Search)
def research_topic(topic: str):
    search = DuckDuckGoSearchRun()
    return search.run(f"{topic} latest research articles")

# Module Structuring Agent
def generate_modules(topic: str):
    llm = ChatOpenAI()
    response = llm.invoke([
        SystemMessage(content="Structure a course with 5 modules on:"),
        HumanMessage(content=topic)
    ])
    return response.content.split("\n")  # Ensure output is a list

# Content Writer Agent
def generate_lesson(module: str):
    llm = OpenAI(gpt-3.5-turbo)
    response = llm.invoke(module)
    return response

# Media Finder Agent
def fetch_relevant_image(topic: str):
    response = openai.Image.create(
        prompt=topic, model="dall-e-2", size="1024x1024", n=1
    )
    return response["data"][0]["url"]

# Quality Control Agent
def validate_content(content: str):
    return "Validated" if "error" not in content else "Needs Review"

@app.get("/generate_course/{topic}")
def generate_course(topic: str):
    research_data = research_topic(topic)
    modules = generate_modules(topic)
    content = [generate_lesson(module) for module in modules]
    media = [fetch_relevant_image(module) for module in modules]
    validation = [validate_content(text) for text in content]
    return {"modules": modules, "content": content, "media": media, "validation": validation}

# requirements.txt
requirements = """
fastapi
uvicorn
langchain
langchain-openai
langchain-community
openai
requests
faiss-cpu
duckduckgo-search
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)
