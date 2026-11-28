!pip install -q langchain langchain-groq langgraph chromadb langchain-text-splitters pydantic

import chromadb
import langchain
import langgraph

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.tools import tool

print("All required libraries imported successfully!")

import os
from google.colab import userdata
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_retries=3
)

response = llm.invoke(
    "Introduce yourself as TripWise AI Travel Assistant."
)

print(response.content)

travel_document = """
TripWise AI - Travel Knowledge Base

Destination: Araku Valley
Category: Nature, Mountains, Adventure
Attractions: Borra Caves, Coffee Museum, Padmapuram Gardens.
Best visiting season: October to March.
Recommended duration: 2 to 3 days.
Estimated hotel cost: Rs 1500 per room per night.
Estimated food cost: Rs 500 per person per day.
Transportation: Train, bus and cab.
Activities: Trekking, sightseeing and coffee plantation visits.

Destination: Visakhapatnam
Category: Beaches, Nature, Heritage
Attractions: RK Beach, Kailasagiri, Submarine Museum,
Rushikonda Beach and Simhachalam Temple.
Best visiting season: October to March.
Recommended duration: 2 to 3 days.
Estimated hotel cost: Rs 2000 per room per night.
Estimated food cost: Rs 600 per person per day.
Transportation: Train, bus, flight and cab.
Activities: Beach visits, sightseeing and museum visits.

Destination: Tirupati
Category: Spiritual, Heritage
Attractions: Tirumala Temple, Sri Padmavathi Temple,
Kapila Theertham and Chandragiri Fort.
Recommended duration: 2 to 3 days.
Estimated hotel cost: Rs 1200 per room per night.
Estimated food cost: Rs 400 per person per day.
Transportation: Train, bus, flight and cab.
Travel tip: Check official temple booking requirements
before planning darshan.

Destination: Hyderabad
Category: Heritage, Food, City Exploration
Attractions: Charminar, Golconda Fort, Salar Jung Museum,
Hussain Sagar and Ramoji Film City.
Best visiting season: October to February.
Recommended duration: 3 to 4 days.
Estimated hotel cost: Rs 2500 per room per night.
Estimated food cost: Rs 700 per person per day.
Transportation: Metro, bus, train, flight and cab.
Popular food: Hyderabadi biryani and Irani chai.

General Travel Guidelines:
Carry valid identification documents.
Check weather conditions before travelling.
Book accommodation in advance during peak seasons.
Carry essential medicines and drinking water.
All budget estimates are approximate.
"""

print("Travel knowledge base created successfully!")
print("Total characters:", len(travel_document))

DESTINATIONS = {
    "araku valley": {
        "hotel": 1500,
        "food": 500,
        "transport": 1000,
        "interests": ["nature", "mountains", "adventure"]
    },

    "visakhapatnam": {
        "hotel": 2000,
        "food": 600,
        "transport": 1200,
        "interests": ["beaches", "nature", "heritage"]
    },

    "tirupati": {
        "hotel": 1200,
        "food": 400,
        "transport": 800,
        "interests": ["spiritual", "heritage"]
    },

    "hyderabad": {
        "hotel": 2500,
        "food": 700,
        "transport": 1500,
        "interests": ["heritage", "food", "city"]
    }
}

print("Available destinations:")

for destination in DESTINATIONS:
    print(destination.title())

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=60
)

chunks = splitter.split_text(travel_document)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
    print("-" * 50)

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

embedding_function = DefaultEmbeddingFunction()

embeddings = embedding_function(chunks)

print("Total embeddings:", len(embeddings))
print("Vector dimensions:", len(embeddings[0]))
print("Sample vector:", embeddings[0][:10])

import chromadb

client = chromadb.PersistentClient(
    path="./tripwise_db"
)

collection = client.get_or_create_collection(
    name="travel_knowledge",
    embedding_function=embedding_function
)

print("TripWise vector database created successfully!")

collection.upsert(
    ids=[f"travel_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=[vector.tolist() for vector in embeddings]
)

print("Travel knowledge stored successfully!")
print("Total documents:", collection.count())

results = collection.query(
    query_texts=["I want to visit mountains and coffee plantations"],
    n_results=2
)

for document in results["documents"][0]:
    print(document)
    print("-" * 50)

questions = [
    "Where can I enjoy beaches?",
    "I want to visit famous temples",
    "Where can I eat famous biryani?",
    "What are the attractions in Araku Valley?"
]

for question in questions:

    results = collection.query(
        query_texts=[question],
        n_results=1
    )

    print("\nQuestion:", question)
    print("Retrieved information:")
    print(results["documents"][0][0])
    print("-" * 50)

from langchain.tools import tool

@tool
def search_travel_knowledge(question: str) -> str:
    """Retrieve travel information about destinations,
    attractions, accommodation, food, transportation
    and travel guidelines."""

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    return "\n\n".join(results["documents"][0])

print(search_travel_knowledge.invoke({
    "question": "What are the famous attractions in Hyderabad?"
}))

import math

@tool
def calculate_trip_budget(
    destination: str, days: int, travelers: int
) -> str:
    """Calculate estimated trip expenses in rupees."""

    destination = destination.lower().strip()

    if destination not in DESTINATIONS:
        return "Destination not available."

    if days < 1 or travelers < 1:
        return "Invalid days or traveler count."

    data = DESTINATIONS[destination]

    rooms = math.ceil(travelers / 2)

    hotel = data["hotel"] * rooms * (days - 1)
    food = data["food"] * days * travelers
    transport = data["transport"] * travelers

    total = hotel + food + transport

    return f"""
Destination: {destination.title()}
Days: {days}
Travelers: {travelers}
Hotel: Rs {hotel}
Food: Rs {food}
Local transport: Rs {transport}
Estimated total: Rs {total}
"""

@tool
def recommend_destination(
    interest: str, budget: int,
    days: int = 2, travelers: int = 1
) -> str:
    """Recommend destinations based on interest and total budget."""

    if budget < 1 or days < 1 or travelers < 1:
        return "Invalid trip details."

    recommendations = []

    for name, data in DESTINATIONS.items():

        cost = (
            data["hotel"] * math.ceil(travelers / 2) * (days - 1)
            + data["food"] * days * travelers
            + data["transport"] * travelers
        )

        if interest.lower().strip() in data["interests"] and cost <= budget:
            recommendations.append(
                f"{name.title()} - Estimated Rs {cost}"
            )

    return "\n".join(recommendations) or "No matching destinations."

@tool
def generate_itinerary(destination: str, days: int) -> str:
    """Generate a day-wise travel itinerary using retrieved knowledge."""

    if destination.lower().strip() not in DESTINATIONS or days < 1:
        return "Invalid destination or duration."

    context = search_travel_knowledge.invoke({
        "question": f"Attractions and activities in {destination}"
    })

    prompt = f"""
    Create a {days}-day itinerary for {destination}.

    Use the following travel information:
    {context}

    Include morning, afternoon and evening activities.
    Do not invent attractions or confirmed bookings.
    """

    return llm.invoke(prompt).content

SYSTEM = """
You are TripWise, an intelligent AI travel assistant.

RULES:
1. Remember traveler details within each thread.
2. Use search_travel_knowledge for destination information.
3. Use calculate_trip_budget for expense calculations.
4. Use recommend_destination for recommendations.
5. Use generate_itinerary for day-wise travel plans.
6. Ask for missing travel details when necessary.
7. Never invent prices, attractions or bookings.
8. Clearly mention that costs are approximate.
9. Respond in a friendly and structured manner.
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=llm,
    tools=[
        search_travel_knowledge,
        calculate_trip_budget,
        recommend_destination,
        generate_itinerary
    ],
    system_prompt=SYSTEM,
    checkpointer=InMemorySaver()
)

print("TripWise AI Agent is ready!")

def chat(message, thread_id="traveler_1"):

    response = agent.invoke(
        {"messages": [
            {"role": "user", "content": message}
        ]},
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    print("TripWise:", response["messages"][-1].content)

chat("My name is Priya. My budget is 10000 rupees.")

chat("Suggest a nature destination for 3 days for 1 person.")

chat("Calculate the budget for Araku Valley for 3 days and 2 people.")

chat("Generate a 3-day itinerary for Araku Valley.")

chat("Do you remember my name and budget?")

chat("My name is Anjali.", "traveler_2")

chat("What is my name?", "traveler_1")

chat("What is my name?", "traveler_2")

print("=" * 50)
print("      WELCOME TO TRIPWISE AI")
print("    Your Intelligent Travel Assistant")
print("=" * 50)

while True:

    message = input("You: ")

    if message.lower().strip() == "exit":
        print("TripWise: Happy travelling!")
        break

    if message.strip():
        chat(message, "live_traveler")

