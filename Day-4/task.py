!pip install -q langchain langchain-groq pydantic

import os
from google.colab import userdata
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_retries=3
)

response = llm.invoke("Welcome customers to GiftVerse.")

print(response.content)

GIFTS = {
    "personalized mug": 299,
    "photo frame": 499,
    "chocolate hamper": 699,
    "customized diary": 399,
    "memory scrapbook": 899,
    "premium gift box": 1499
}

for gift, price in GIFTS.items():
    print(f"{gift.title()}: Rs. {price}")

OCCASIONS = {
    "birthday": [
        "personalized mug",
        "chocolate hamper",
        "premium gift box"
    ],
    "anniversary": [
        "photo frame",
        "memory scrapbook",
        "premium gift box"
    ],
    "friendship": [
        "customized diary",
        "personalized mug",
        "memory scrapbook"
    ],
    "graduation": [
        "customized diary",
        "photo frame",
        "premium gift box"
    ]
}

print("Gift catalogue and occasion mapping ready!")

from typing import Literal
from pydantic import BaseModel, Field
from datetime import date

class GiftOrder(BaseModel):

    name: str = Field(min_length=1)

    gift: Literal[
        "personalized mug",
        "photo frame",
        "chocolate hamper",
        "customized diary",
        "memory scrapbook",
        "premium gift box"
    ]

    quantity: int = Field(ge=1)
    occasion: str
    delivery_date: date
    delivery_address: str = Field(min_length=5)
    personal_message: str = "No message"

order = GiftOrder(
    name="Priya",
    gift="photo frame",
    quantity=2,
    occasion="birthday",
    delivery_date="2026-12-05",
    delivery_address="Kakinada, Andhra Pradesh",
    personal_message="Happy Birthday!"
)

print(order.model_dump(mode="json"))

from langchain.tools import tool

@tool
def get_gift_price(gift: str, quantity: int = 1) -> str:
    """Calculate the total price of a gift."""

    gift = gift.lower().strip()

    if gift not in GIFTS:
        return "Gift not available in our catalogue."

    if quantity < 1:
        return "Quantity must be at least 1."

    total = GIFTS[gift] * quantity

    return f"{quantity} x {gift} = Rs. {total}"

@tool
def check_availability(gift: str) -> str:
    """Check whether a gift is listed in our catalogue."""

    gift = gift.lower().strip()

    if gift in GIFTS:
        return f"{gift.title()} is available in our catalogue."

    return f"{gift.title()} is not in our catalogue."

@tool
def recommend_gifts(occasion: str, budget: int) -> str:
    """Recommend gifts based on occasion and budget."""

    occasion = occasion.lower().strip()

    if budget < 1:
        return "Please enter a valid budget."

    items = OCCASIONS.get(occasion, list(GIFTS))

    recommendations = []

    for gift in items:
        if GIFTS[gift] <= budget:
            recommendations.append(
                f"{gift.title()} - Rs. {GIFTS[gift]}"
            )

    if not recommendations:
        return "No matching gifts within your budget."

    return "\n".join(recommendations)

print(get_gift_price.invoke({
    "gift": "photo frame",
    "quantity": 2
}))

print(check_availability.invoke({
    "gift": "teddy bear"
}))

print(recommend_gifts.invoke({
    "occasion": "birthday",
    "budget": 1000
}))

import os
import csv

ORDERS_FILE = "giftverse_orders.csv"

@tool
def save_gift_order(
    name: str,
    gift: str,
    quantity: int,
    occasion: str,
    delivery_date: str,
    delivery_address: str,
    personal_message: str = "No message"
) -> str:
    """Validate and save a confirmed gift order to CSV."""

    try:
        order = GiftOrder(
            name=name.strip(),
            gift=gift.lower().strip(),
            quantity=quantity,
            occasion=occasion,
            delivery_date=delivery_date,
            delivery_address=delivery_address.strip(),
            personal_message=personal_message
        )

        if order.delivery_date < date.today():
            return "Delivery date cannot be in the past."

        total = GIFTS[order.gift] * order.quantity

        file_exists = (
            os.path.exists(ORDERS_FILE)
            and os.path.getsize(ORDERS_FILE) > 0
        )

        with open(
            ORDERS_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "Name", "Gift", "Quantity",
                    "Occasion", "Delivery Date",
                    "Address", "Personal Message",
                    "Total"
                ])

            writer.writerow([
                order.name,
                order.gift,
                order.quantity,
                order.occasion,
                order.delivery_date,
                order.delivery_address,
                order.personal_message,
                total
            ])

        return f"Order saved successfully! Total: Rs. {total}"

    except Exception as e:
        return f"Could not save order: {e}"

from datetime import timedelta

print(save_gift_order.invoke({
    "name": "Priya",
    "gift": "photo frame",
    "quantity": 2,
    "occasion": "birthday",
    "delivery_date": str(date.today() + timedelta(days=7)),
    "delivery_address": "Kakinada, Andhra Pradesh",
    "personal_message": "Happy Birthday!"
}))

import pandas as pd

df = pd.read_csv("giftverse_orders.csv")

display(df)

SYSTEM = """
ROLE:
You are a friendly AI assistant for GiftVerse,
a personalized gift shop.

RULES:
1. Use get_gift_price to calculate prices.
2. Use check_availability to check products.
3. Use recommend_gifts to suggest gifts based
   on occasion and budget.
4. Never invent products or prices.
5. Collect customer name, gift, quantity,
   occasion, delivery date and address.
6. Ask whether the customer wants a personal message.
7. Use YYYY-MM-DD format for delivery dates.
8. Show the complete order summary and total.
9. Ask for confirmation before saving an order.
10. Use save_gift_order only after confirmation.
11. Avoid saving duplicate orders.
12. Confirm success only when the saving tool succeeds.

FORMAT:
Be friendly, helpful and concise.
Respond in English or Telugu based on the customer.
"""

from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[
        get_gift_price,
        check_availability,
        recommend_gifts,
        save_gift_order
    ],
    system_prompt=SYSTEM
)

print("GiftVerse AI Agent created successfully!")

history = []

def chat(user_input):

    history.append({
        "role": "user",
        "content": user_input
    })

    response = agent.invoke({
        "messages": history
    })

    history[:] = response["messages"]

    answer = history[-1].content

    print("GiftVerse:", answer)

chat("Suggest birthday gifts under 1000 rupees")

chat("What is the price of 3 personalized mugs?")

chat("Do you have teddy bears?")

history.clear()

chat("My name is Priya")
chat("I want a gift for my friend's birthday")
chat("My budget is 1000 rupees")
chat("What is my name and budget?")

history.clear()

chat(f"""
My name is Anjali.
I want 2 chocolate hampers for a birthday.
Delivery date: {date.today() + timedelta(days=7)}
Delivery address: Kakinada, Andhra Pradesh.
Personal message: Happy Birthday Dear Friend!
""")

chat("Yes, confirm my order.")

df = pd.read_csv(ORDERS_FILE)
display(df)

history.clear()

print("=" * 50)
print("       WELCOME TO GIFTVERSE AI")
print("    Personalized Gift Shop Assistant")
print("=" * 50)

print("Type 'exit' to end the conversation.\n")

while True:

    user_input = input("You: ")

    if user_input.lower().strip() == "exit":
        print("GiftVerse: Thank you for visiting!")
        break

    if not user_input.strip():
        continue

    chat(user_input)

from google.colab import files

files.download("giftverse_orders.csv")

