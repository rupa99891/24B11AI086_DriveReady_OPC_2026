import os
import csv
from pathlib import Path
from getpass import getpass
from langchain_groq import ChatGroq

# API key configuration
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass("Enter Groq API key: ")

# Business details
business = {
    "name": "TripWise Travel Services",
    "service": "Domestic travel packages",
    "budget_price": 5000,
    "standard_price": 10000,
    "premium_price": 20000,
    "working_hours": "9 AM to 6 PM",
    "cancellation": "48 hours before departure"
}

# Original system prompt
SYSTEM = f"""
Role: You are a travel assistant for {business['name']}.

Facts:
- We provide {business['service']}.
- Budget package starts at Rs. {business['budget_price']}.
- Standard package starts at Rs. {business['standard_price']}.
- Premium package starts at Rs. {business['premium_price']}.
- Working hours: {business['working_hours']}.
- Cancellation is allowed up to {business['cancellation']}.

Rules:
- Be polite and helpful.
- Answer only travel-related questions.
- Do not invent business information.

Format:
- Keep answers short and clear.
"""

# Improved system prompt
IMPROVED_SYSTEM = SYSTEM + """
Important: Preserve the exact words, numerical values,
and time formats from the business facts when answering.
"""

# Automated testing
def run_tests(model, system_prompt):
    passed = 0
    total = 0

    csv_path = Path(__file__).resolve().parent / "tests.csv"

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            question = row["question"]
            expected = row["must_include"]

            response = model.invoke([
                ("system", system_prompt),
                ("human", question)
            ])

            answer = response.content
            total += 1

            print("Question:", question)
            print("AI Answer:", answer)
            print("Expected:", expected)

            if expected.lower() in answer.lower():
                passed += 1
                print("Result: PASS")
            else:
                print("Result: FAIL")

            print("-" * 50)

    pass_rate = (passed / total) * 100

    print("Total Tests:", total)
    print("Passed:", passed)
    print("Failed:", total - passed)
    print("Pass Rate:", pass_rate, "%")

    return pass_rate


def main():
    # Initialize LLM with temperature 0
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    print("\n===== BASELINE TESTING =====")
    baseline_rate = run_tests(llm, SYSTEM)

    print("\n===== IMPROVED PROMPT TESTING =====")
    improved_rate = run_tests(llm, IMPROVED_SYSTEM)

    # Temperature 1.0 testing
    llm_high = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=1.0
    )

    print("\n===== TEMPERATURE 1.0 - RUN 1 =====")
    run1_rate = run_tests(llm_high, IMPROVED_SYSTEM)

    print("\n===== TEMPERATURE 1.0 - RUN 2 =====")
    run2_rate = run_tests(llm_high, IMPROVED_SYSTEM)

    # Performance comparison
    print("\n===== FINAL COMPARISON =====")
    print("Baseline:", baseline_rate, "%")
    print("Improved:", improved_rate, "%")
    print("Improvement:", improved_rate - baseline_rate,
          "percentage points")
    print("Temperature 1.0 Run 1:", run1_rate, "%")
    print("Temperature 1.0 Run 2:", run2_rate, "%")

    if run1_rate == run2_rate:
        print("Pass rates are the same.")
    else:
        print("Pass rates are different.")

    # Agent or Not classification
    jobs = [
        {
            "job": "Calculate total travel package cost",
            "type": "Plain Code",
            "reason": "Uses fixed prices and arithmetic."
        },
        {
            "job": "Answer customer travel questions",
            "type": "One LLM Call",
            "reason": "Generates answers using a system prompt."
        },
        {
            "job": "Plan a complete personalized trip",
            "type": "Agent",
            "reason": "Chooses tools, compares options and adjusts plans."
        }
    ]

    print("\n===== AGENT OR NOT =====")

    for job in jobs:
        print("Job:", job["job"])
        print("Type:", job["type"])
        print("Reason:", job["reason"])
        print("-" * 50)


if __name__ == "__main__":
    main()
