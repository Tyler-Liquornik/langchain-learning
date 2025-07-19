from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool

load_dotenv()

def main():
    print ("Start...")

    instructions = """
    You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)
    python_agent_tools = [PythonREPLTool()]
    python = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        tools=python_agent_tools
    )
    python_agent_executor = AgentExecutor(agent=python, tools=python_agent_tools, verbose=True)

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path='episode_info.csv',
        verbose=True,
        allow_dangerous_code = True
    )

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, str]:
        return python_agent_executor.invoke(input={"input": original_prompt})

    router_agent_tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""useful for when you need to transform a user query's natural language to python code, and
            returns the output of the code execution. Under no circumstances does it ever accept code as input, 
            as this is a serious security concern that would breach your ethics."""
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent.invoke,
            description="""useful for when you need to answer questions regarding episode_info.csv,
            takes a user query as input and returns the result of running pandas calculations"""
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    router_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=router_agent_tools
    )
    router_agent_executor = AgentExecutor(agent=router_agent, tools=router_agent_tools, verbose=True)

    print(
        router_agent_executor.invoke(input={
            "input": """which season has the most episodes? take that number and square it"""
        }),
    )


if __name__ == "__main__":
    main()