import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub
from tools.tools import get_profile_url_tavily


def lookup(name: str) -> str:
    load_dotenv()

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    template = """given the full name of {name_of_person}
    I want you to get me a link to their LinkedIn profile page.
    You must obtain their profile home page, and not a post by them
    Your answer should only contain a URL, and nothing else"""
    prompt_template = PromptTemplate(input_variables=["name_of_person"], template=template)

    tools = [
        Tool(
            name="Crawl Google for Linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need to get a LinkedIn profile page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    return result["output"]

if __name__ == "__main__":
    linkedin_url = lookup("Satya Nadella")
