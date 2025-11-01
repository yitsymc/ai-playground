# --- vendor_agent_llama_langchain_memory.py ---
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from difflib import get_close_matches
from typing import Optional

# --- Mock Vendor Data ---
VENDORS = {
    "Solaris Industries": {
        "email": "sales@solarisindustries.com",
        "phone": "+34 91 555 8811",
        "country": "Spain",
        "avg_delivery_days": 4,
        "on_time_rate": 98,
        "open_orders": 1,
        "avg_invoice_amount": 5400.00,
        "last_purchase_date": "2024-10-18",
        "satisfaction": 4.8,
        "quality_score": 9.5,
        "contract_renewal_date": "2025-03-01",
    },
    "GreenTech Supplies": {
        "email": "info@greentech.eu",
        "phone": "+49 30 1234 5678",
        "country": "Germany",
        "avg_delivery_days": 9,
        "on_time_rate": 72,
        "open_orders": 3,
        "avg_invoice_amount": 3200.00,
        "last_purchase_date": "2024-09-22",
        "satisfaction": 3.4,
        "quality_score": 7.1,
        "contract_renewal_date": "2025-01-15",
    },
    "EcoParts Ltd": {
        "email": "parts@ecoparts.uk",
        "phone": "+44 161 555 9082",
        "country": "United Kingdom",
        "avg_delivery_days": 5,
        "on_time_rate": 89,
        "open_orders": 2,
        "avg_invoice_amount": 2750.00,
        "last_purchase_date": "2024-10-02",
        "satisfaction": 4.1,
        "quality_score": 8.3,
        "contract_renewal_date": "2025-02-10",
    },
}

# --- Vendor Logic ---
def find_vendor_match(query: str) -> Optional[str]:
    """Find the closest vendor name match."""
    names = list(VENDORS.keys())
    match = get_close_matches(query, names, n=1, cutoff=0.5)
    return match[0] if match else None


def analyze_vendor_health(vendor: str) -> str:
    """Provide a summary of a vendor’s performance."""
    data = VENDORS.get(vendor)
    if not data:
        return f"No data found for vendor '{vendor}'."

    score = 0
    if data["on_time_rate"] > 90: score += 2
    if data["satisfaction"] > 4.5: score += 2
    if data["quality_score"] > 8: score += 2
    if data["avg_delivery_days"] <= 5: score += 1
    if data["open_orders"] <= 2: score += 1

    status = (
        "excellent" if score >= 7 else
        "solid" if score >= 5 else
        "moderate" if score >= 3 else
        "risky"
    )

    insights = f"Vendor **{vendor}** ({data['country']}) appears to be in **{status} condition**.\n"
    insights += f"- On-time deliveries: {data['on_time_rate']}%\n"
    insights += f"- Satisfaction: {data['satisfaction']}/5\n"
    insights += f"- Quality score: {data['quality_score']}/10\n"
    insights += f"- Avg. delivery time: {data['avg_delivery_days']} days\n"
    insights += f"- Next contract renewal: {data['contract_renewal_date']}\n"

    if status in ["moderate", "risky"]:
        insights += "\n⚠️ Consider reviewing contract terms or requesting quality improvements."
    elif status == "solid":
        insights += "\n✅ Generally reliable. Maintain the partnership and monitor next renewal."
    else:
        insights += "\n🌟 Excellent partner — consider long-term agreements or preferred supplier status."

    return insights


# --- Setup Model & Prompt ---
llm = OllamaLLM(model="llama3.2:3b-instruct-q4_K_M", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Vendor Evaluation Assistant for Odoo. "
     "You help users analyze supplier reliability and suggest actions. "
     "If you recognize a vendor name, show its evaluation. "
     "Otherwise, answer conversationally using context."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

def chat_with_agent(user_input: str) -> str:
    vendor = find_vendor_match(user_input)
    if vendor:
        analysis = analyze_vendor_health(vendor)
        memory.save_context({"input": user_input}, {"output": analysis})
        return analysis
    else:
        return chain.predict(input=user_input)


if __name__ == "__main__":
    print("💬 Vendor Evaluation Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        reply = chat_with_agent(user_input)
        print("Agent:", reply)
