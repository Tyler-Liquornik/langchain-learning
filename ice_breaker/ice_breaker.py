from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

def ice_break_with(name: str):
    linkedin_url = linkedin_lookup_agent(name=name)
    linkedin_profile = scrape_linkedin_profile(linkedin_profile_url=linkedin_url)

    summary_template = """
          given the information {information} about a person, I want you to create:
          1. a short summary
          2. two interesting facts about them
      """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

    chain = summary_prompt_template | llm
    res = chain.invoke(input={"information": linkedin_profile})
    print(res)


if __name__ == "__main__":
    load_dotenv()
    person_name = input("Break the Ice With: ")
    print(ice_break_with(name=person_name))