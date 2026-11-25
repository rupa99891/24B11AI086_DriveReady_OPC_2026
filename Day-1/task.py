
import os
from getpass import getpass
from langchain_groq import ChatGroq

# Configure API key securely
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass("Enter Groq API key: ")

# Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# First API call
response = llm.invoke(
    "Explain how Large Language Models work in 5 simple points."
)

print("AI Response:")
print(response.content)

# Interactive AI Study Assistant
question = input("\nAsk your AI Study Assistant: ")

response = llm.invoke(question)

print("\nAI Response:")
print(response.content)
