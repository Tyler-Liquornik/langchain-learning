from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()

    summary_template = """
        given the information {information} about a person, I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

    chain = summary_prompt_template | llm

    linkedin_data = scrape_linkedin_profile("https://www.linkedin.com/in/tyler-liquornik/")

    res = chain.invoke(input={"information": linkedin_data})

    print(res)