from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from llm import llm, tools

load_dotenv()

SYSTEM_PROMPT="""
You are a helpful assistant that can use tools to answer questions.
"""

def reason_node(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning
    """
    response = llm.invoke([{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]])
    return {"messages": [response]}

act_node = ToolNode(tools)