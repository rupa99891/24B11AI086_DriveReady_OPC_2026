
import os
import json
from getpass import getpass
from langchain_groq import ChatGroq

# 1. API key configuration
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass("Enter Groq API key: ")

# 2. Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)

# 3. System Prompt Engineering
SYSTEM_PROMPT = """
Role: You are an AI Travel Assistant.

Facts:
- We provide domestic travel packages.
- Budget package starts at Rs. 5000.
- Standard package starts at Rs. 10000.
- Premium package starts at Rs. 20000.

Rules:
- Be polite and helpful.
- Answer only travel-related questions.
- Do not invent prices or booking details.
- Suggest packages based on the user's budget.

Format:
- Give clear and simple answers.
- Keep responses under 60 words.
"""

messages = [
    ("system", SYSTEM_PROMPT),
    ("human", "Suggest a travel package under Rs. 8000.")
]

response = llm.invoke(messages)

print("\n--- System Prompting ---")
print(response.content)

# 4. Zero-shot Prompting
messages = [
    ("system", """
    Classify travel requests as Booking, Enquiry
    or Cancellation. Return only the category name.
    """),
    ("human", "I want to cancel my Goa trip.")
]

response = llm.invoke(messages)

print("\n--- Zero-shot Prompting ---")
print("Category:", response.content)

# 5. Few-shot Prompting and Classification
def classify_request(user_message):
    messages = [
        ("system", """
        Classify travel requests as Booking, Enquiry
        or Cancellation. Return only the category name.
        """),

        ("human", "Book a trip to Kerala."),
        ("ai", "Booking"),

        ("human", "What is the cost of a Goa trip?"),
        ("ai", "Enquiry"),

        ("human", "Cancel my hotel booking."),
        ("ai", "Cancellation"),

        ("human", user_message)
    ]

    response = llm.invoke(messages)
    return response.content.strip()


print("\n--- Few-shot Prompting ---")

user_message = input("Enter your travel request: ")

category = classify_request(user_message)

print("Request Category:", category)

# 6. JSON Data Extraction
print("\n--- JSON Data Extraction ---")

travel_request = input("Enter your travel details: ")

messages = [
    ("system", """
    Extract travel details from the user message.

    Return ONLY valid JSON with these keys:
    name, destination, travelers, days, budget.

    Use integers for travelers, days and budget.
    Use null if information is missing.
    Do not include markdown or explanations.
    """),

    ("human", travel_request)
]

response = llm.invoke(messages)

reply = response.content

print("JSON Response:")
print(reply)

# 7. JSON Parsing and Exception Handling
try:
    travel = json.loads(reply)

    print("\n--- Travel Details ---")
    print("Name:", travel["name"])
    print("Destination:", travel["destination"])
    print("Travelers:", travel["travelers"])
    print("Days:", travel["days"])
    print("Budget:", travel["budget"])

except json.JSONDecodeError:
    print("Invalid JSON response!")
    print(reply)

except KeyError as e:
    print("Missing field:", e)
