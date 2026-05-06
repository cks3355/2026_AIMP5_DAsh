import os
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = AzureOpenAI(
    azure_endpoint=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    api_version="2024-12-01-preview",
)

resp = client.chat.completions.create(
    model="gpt-5",              # AOAI deployment 이름
    messages=[{"role": "user", "content": "Multi Agent Architecture 기술에 대해 설명해줘"}],
)
print(resp.choices[0].message.content)