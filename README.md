# ai-playground
Experimental space for testing AI workflows: reasoning agents, custom tools, and RAG integrations, etc.

## 🧩 What’s inside
- 🧩 ReAct-style agents with local LLMs  
- 🔍 Retrieval-Augmented Generation (RAG) for financial and textual data  
- ⚙️ Custom LangChain tools connected to APIs (Yahoo Finance, DuckDuckGo, etc.)  
- 📊 Streamlit dashboards for real-time visualization  

Each script is a small, self-contained experiment you can run locally with **Python**.

## Scripts
- **Financial Fundamentals Agent (`langchain_finance_agent.py`)**: This LangChain ReAct agent connects **Ollama** (Llama 3.2) with **Yahoo Finance** through `yfinance` to answer analytical questions about a company’s fundamentals — using tool-based reasoning.
  * Fetches real metrics like **P/E**, **EPS**, **ROE**, and **Debt/Equity**.
  * Uses LangChain’s **Tool + ReAct** agent architecture.
  * Lets the model reason over the retrieved data.  
  * Runs **fully locally** (no external API costs).
 
## Setup
1. Download the script.
2. Install the requirements: `pip install -r SCRIPT_NAME_requirements.txt`.
3. You also need Ollama installed locally and a model like `llama3.2:3b-instruct-q4_K_M` pulled.
