import yfinance as yf
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool


# Tools
def get_pe_ratio(symbol: str):
    """Fetch the Price-to-Earnings (P/E) ratio."""
    try:
        stock = yf.Ticker(symbol)
        pe = stock.info.get("trailingPE", None)
        return (
            f"{symbol.upper()} P/E ratio: {round(pe, 2)}"
            if pe
            else f"No P/E data for {symbol.upper()}."
        )
    except Exception as e:
        return f"Error fetching P/E ratio for {symbol.upper()}: {e}"


def get_eps(symbol: str):
    """Fetch the Earnings per Share (EPS)."""
    try:
        stock = yf.Ticker(symbol)
        eps = stock.info.get("trailingEps", None)
        return (
            f"{symbol.upper()} EPS: {round(eps, 2)}"
            if eps
            else f"No EPS data for {symbol.upper()}."
        )
    except Exception as e:
        return f"Error fetching EPS for {symbol.upper()}: {e}"


def get_pb_ratio(symbol: str):
    """Fetch the Price-to-Book (P/B) ratio."""
    try:
        stock = yf.Ticker(symbol)
        pb = stock.info.get("priceToBook", None)
        return (
            f"{symbol.upper()} P/B ratio: {round(pb, 2)}"
            if pb
            else f"No P/B data for {symbol.upper()}."
        )
    except Exception as e:
        return f"Error fetching P/B ratio for {symbol.upper()}: {e}"


def get_roe(symbol: str):
    """Fetch Return on Equity (ROE)."""
    try:
        stock = yf.Ticker(symbol)
        roe = stock.info.get("returnOnEquity", None)
        return (
            f"{symbol.upper()} ROE: {round(roe*100, 2)}%"
            if roe
            else f"No ROE data for {symbol.upper()}."
        )
    except Exception as e:
        return f"Error fetching ROE for {symbol.upper()}: {e}"


def get_debt_to_equity(symbol: str):
    """Fetch the Debt-to-Equity ratio."""
    try:
        stock = yf.Ticker(symbol)
        debt_equity = stock.info.get("debtToEquity", None)
        return (
            f"{symbol.upper()} Debt/Equity: {round(debt_equity, 2)}"
            if debt_equity
            else f"No Debt/Equity data for {symbol.upper()}."
        )
    except Exception as e:
        return f"Error fetching Debt/Equity for {symbol.upper()}: {e}"


tools = [
    Tool(
        name="GetPERatio",
        func=get_pe_ratio,
        description="Fetches the company's P/E ratio.",
    ),
    Tool(
        name="GetEPS",
        func=get_eps,
        description="Fetches the company's Earnings per Share (EPS).",
    ),
    Tool(
        name="GetPBRatio",
        func=get_pb_ratio,
        description="Fetches the company's Price-to-Book (P/B) ratio.",
    ),
    Tool(
        name="GetROE",
        func=get_roe,
        description="Fetches the company's Return on Equity (ROE).",
    ),
    Tool(
        name="GetDebtToEquity",
        func=get_debt_to_equity,
        description="Fetches the company's Debt-to-Equity ratio.",
    ),
]


llm = OllamaLLM(model="llama3.2:3b-instruct-q4_K_M", temperature=0.2)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


ticker = input("Enter the company ticker symbol (e.g., NVDA, AAPL, MSFT): ").upper()
print("\nChoose a question to ask about", ticker)
questions = [
    f"Based on fundamentals, is {ticker} currently overvalued or undervalued?",
    f"Is {ticker} an efficient business based on its ROE?",
    f"Does {ticker} carry too much debt compared to its equity?",
    f"Summarize {ticker}'s financial health in 3 sentences.",
    f"What are the main financial strengths and weaknesses of {ticker}?",
]

for i, q in enumerate(questions, start=1):
    print(f"{i}. {q}")

choice = input("\nEnter the number of the question you want to ask (1-5): ")

try:
    selected_question = questions[int(choice) - 1]
except (IndexError, ValueError):
    print("Invalid choice. Exiting.")
    exit()

print("\n🤖 Thinking...\n")
response = agent.run(selected_question)
print("\n--- AGENT RESPONSE ---\n")
print(response)
