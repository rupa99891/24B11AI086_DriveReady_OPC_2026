# Day 2 - Prompt Engineering, JSON and Python Basics

## Project: AI Travel Assistant

An interactive AI-powered travel assistant that provides travel package suggestions, classifies user requests, and extracts structured travel information from natural language.

## Objectives
- Understand prompt engineering techniques.
- Implement zero-shot and few-shot prompting.
- Classify travel requests using an LLM.
- Extract and parse structured JSON data.
- Implement Python exception handling.

## Technologies Used
- Python
- Google Colab
- Groq API
- LangChain
- GPT-OSS 120B
- JSON

## Features
- Travel package recommendations using sample prices.
- System prompting using Role, Facts, Rules and Format.
- Zero-shot and few-shot prompting.
- Dynamic travel request classification.
- Classification into Booking, Enquiry and Cancellation.
- Structured JSON data extraction.
- JSON parsing and exception handling.

## Sample Input
My name is Riya. I want to visit Araku for 3 days with 4 people. My budget is 20000 rupees.

## Sample JSON Output
```json
{
  "name": "Riya",
  "destination": "Araku",
  "travelers": 4,
  "days": 3,
  "budget": 20000
}
```

## Key Learnings
- System prompts and prompt engineering.
- Zero-shot and few-shot learning.
- Python functions, lists, tuples and dictionaries.
- Natural language classification.
- JSON extraction and parsing.
- Exception handling using try and except.
- Secure API key management.

## How to Run
1. Install the dependency: pip install langchain-groq
2. Run: python task.py
3. Enter your Groq API key if prompted.
4. Enter a travel request for classification.
5. Enter travel details for JSON extraction.

## Outcome
Successfully implemented and tested an AI Travel Assistant using prompt engineering and structured data extraction.

## Note
Travel package prices are sample values used for educational purposes. This application does not perform actual bookings.
