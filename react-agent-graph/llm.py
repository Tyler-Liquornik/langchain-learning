from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

@tool
def triple(num: float) -> float:
    """Multiply a number by 3"""
    return num * 3

tools = [triple, TavilySearchResults(max_results=1)]

llm = (ChatOpenAI(temperature=0, model="gpt-4o-mini")
       .bind_tools(tools))

