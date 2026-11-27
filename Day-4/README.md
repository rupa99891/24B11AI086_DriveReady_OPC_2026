# Day 04 – AI Agents, Tools & Structured Data

## Overview

Day 04 focuses on developing intelligent AI agents using custom tools, function calling, Pydantic validation and data storage.

## Concepts Covered

- Custom tools using the `@tool` decorator
- Function calling in AI agents
- Pydantic models and data validation
- Google Sheets integration
- CSV file handling
- System prompts
- Conversation history
- Interactive AI agents

## Classroom Practice – Sweet Crumbs Bakery

Developed an AI-powered bakery assistant that handles product enquiries, price calculations, business hours and customer orders using Google Sheets.

## Assignment – GiftVerse AI Agent

GiftVerse is an intelligent personalized gift shop assistant that helps customers discover suitable gifts based on their occasion and budget.

### Features

- Occasion-based gift recommendations
- Budget-based gift filtering
- Gift availability checking
- Automatic price calculation
- Personalized gift messages
- Customer order validation
- Order confirmation
- CSV-based order storage
- Conversational memory

### Custom Tools

| Tool | Functionality |
|---|---|
| `get_gift_price` | Calculates total gift price |
| `check_availability` | Checks gift catalogue availability |
| `recommend_gifts` | Suggests gifts based on occasion and budget |
| `save_gift_order` | Validates and stores customer orders |

## Technologies Used

- Python
- LangChain
- Groq LLM
- Pydantic
- Pandas
- Google Sheets
- CSV
- Google Colab

## Files

- `Day_04.ipynb` – Classroom practice
- `task.py` – GiftVerse AI Agent assignment

## Learning Outcomes

- Developed AI agents with custom tools.
- Implemented structured data validation.
- Integrated AI agents with external data storage.
- Built a rule-based recommendation system.
- Managed multi-turn conversations.
- Explored AI-driven business automation.

## Limitations

- Gift availability is based on a predefined catalogue.
- Conversation history is temporary.
- Order confirmation and duplicate prevention are prompt-based.
- The assignment was developed in Google Colab.

---

**Drive Ready – AI Agents | OPC Initiative 2026**

**Learn. Build. Implement. Grow.**
