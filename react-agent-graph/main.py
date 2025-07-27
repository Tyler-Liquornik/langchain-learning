from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from nodes import reason_node, act_node

load_dotenv()

REASON = "reason"
ACT = "act"

# If the last message is not a tool call, end, otherwise act
def should_continue(state: MessagesState) -> str:
    if not state["messages"][-1].tool_calls: return END
    return ACT

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node(REASON, reason_node)
graph.add_node(ACT, act_node)
graph.add_edge(START, REASON)
graph.add_edge(ACT, REASON)
graph.add_conditional_edges(REASON, should_continue, {
    END: END, # Mapping should_continue outputs to node names
    ACT: ACT  # We use the same names, so the key & val is the same
})

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    print("Hello LangGraph world!")
    res = app.invoke({"messages": [HumanMessage(content="What is triple the current temperature in Toronto?")]})
    print(res["messages"][-1].content)