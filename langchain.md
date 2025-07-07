  
## Learning Goal  
  
The goal for this course is to be able to build AI applications in two domains: AI Agents, and Retrieval Augmented Generation. I would like to gain an understanding of the mechanisms that drive these applications under the hood, which means being adept at Prompt Engineering, and understanding the LangChain Ecosystem stack, like using LangSmith for tracing, and LangGraph for workflow engineering. In addition to all this, I'll cover how to create production ready LangChain apps, with enterprise concerns like testing, logging & monitoring, security, and more.  
  
Theres also gonna be some python basics in here, as I'm relatively new to building full fledged python applications that aren't just small scripts, and am writing from the perspective of a developer with more experience writing Java and JavaScript applications.  
  
> Note: This guide is updated for LangChain 0.3.3  
  
## What is LangChain?  
  
LangChain is an open source framework that simplifies the process of building LLM applications, with a set of useful abstractions that make it so we don't need understand the inner workings of LLMs, but rather use them as a blackbox for software engineering.  
  
One example of this is that it lets us switch between different LLMs extremely easily, using the same interface for every model. This helps very easily prevent vendor lock with object design around one specific model.  
  
```python  
from langchain_anthropic import ChatAnthropic  
from langchain_openai import ChatOpenAI  
from langchain_ollama import ChatOllama  
  
llm = ChatAnthropic(model='claude-3-opus-20240229')  
llm = ChatOpenAI(model="gpt-4o", temperature = 0)  
llm = ChatOllama(model="llama3.2")  
```  
  
Another topic is injecting external text into text prompts. This comes up often as we know, in things like context augmented generation, and we'll see it in RAG too.  
  
```python  
from langchain_core.prompts import PromptTemplate  
  
prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")  
prompt_template.invoke({"topic": "cats"})  
```  
  
There are also document loaders, for documents in many forms, like Notion databases, PDFs, emails, and more  
  
```python  
from langchain_community.document_loaders import NotionDirectoryLoader  
from langchain_community.document_loaders import PyPDFLoader  
from langchain_community.document_loaders import UnstructuredEmailLoader  
  
notion_loader = NotionDirectoryLoader("Notion_DB")  
notion_docs = notion_loader.load()  
  
pdf_loader = PyPDFLoader("your_file.pdf")  
pdf_docs = pdf_loader.load()  
  
email_loader = UnstructuredEmailLoader("example-email.eml")  
email_docs = email_loader.load()  
```  
  
There are many more features on top of these 3, these just scratch the tip of the iceberg!  
  
## Setting Up Your Environment  
  
Get Started by installing `pipenv` to use as our virtual environment and package manager. If you don't have it already, simply:  
  
```sh  
brew install pipenv
```  
  
And then in PyCharm, be sure to select the binary for pipenv under *Settings => Project: <project_name> => Python Interpreter*  
  
Startup the environment with `pipenv shell`, which either locates or creates your virtual-env, starts up a shell inside it, and configures a lot of the base environment variables for the env like `PATH`, `VIRTUAL_ENV`, and `PYTHONPATH` so things we run now properly execute inside the env.  
  
Now that the environment is setup, to get started with LangChain, lets install some dependencies to start us off. Start with the framework itself `pipenv install langchain`. In addition, we'll `pipenv install langchain-openai` to get OpenAI's third party LangChain integration package, and `pipenv install langchain-community` for a lot of useful community contributed code. Also `pipenv install langchainhub`, which includes a lot of community written prompts that you can easily snag from that already have good prompt engineering behind them.   
  
Aside from LangChain packages, there are some other packages to install too. `pipenv install python-dotenv` will give us easy access to env vars in `.env`, which we'll use often for things like LLM API keys. It has a dead simple API:  
  
```python  
from dotenv import load_dotenv  
import os  
  
if __name__ == "__main__":  
     # Call this once at the start    
     load_dotenv()    
     # API Key value accessible assuming its in .env   
     print(os.environ['API_KEY'])  
```  
  
As a bonus package, `pipenv install black`, for a linter, which you can just run right away with `black .` to keep our code neat.  
  
A lot of the reason for all these different packages is the refactoring  through diverse contributions over time since LangChain started. For example before `langchain-openai`, you had to use the whole OpenAI SDK separately, but now its more unified with official integration packages that each LLM vendor maintains directly with LangChain.  
  
## Prompts and LLM APIs  
  
Lets first start with two interfaces, `PromptTemplate`, and `ChatOpenAI`, which we can grab with:  
```python  
from langchain_core.prompts import PromptTemplate  
from langchain_openai import ChatOpenAI  
```  
  
Consider the prompt: `"I want you to write a cool funny jingle for a {product} product`. The jingle should be 1 sentence." We could use this to help the user generate some content, and we'd inject something like `product = sports shoes`, `product = piano`, or `product = cat food`.  
  
`PromptTemplate` wraps a prompt by allowing it to take input parameters, changing the outcome of a prompt sent to our LLM. We pass in two key parameters to start using `PromptTemplate`, `input_variables: list[str]` which is the list of `{placeholders}` to replace and `template: str` which is the prompt containing the placeholders.  
  
We use `PromptTemplate` among other abstractions to make LLM queries in what we call a **chain**. A chain is simply a way of combining together different steps in a workflow using AI across different steps, to transform an input into an output.  
  
![](images/langchain-chain.png)  
  
We can create our first chain by piping our `PromptTemplate` into `ChatOpenAI`, which takes in a `temperature: int` from 0 to 1 for creativity, 1 meaning its more creative / hallucinates more. It also takes in `model: str` to give the name of the LLM, for us `gpt-3.5-turbo` since its cheap. `ChatOpenAI` already looks for env var `OPENAI_API_KEY` and handles creating the OpenAI client to hit their API, meaning no need to manually `load_dotenv()` or handle any of that annoying work, just populate `OPENAI_API_KEY` in `.env` and the runtime is configured to read it.  
  
For our chain, LangChain conveniently overloads the `|` operator which we use to create a chain just like bash piping, anything runnable can be used on either side of the pipe operator which includes prompt templates, models, and other abstractions we'll discover later -- but also even if you pipe in a raw string LangChain will automatically wrap it with the necessary functionality so that works too..  
  
Finally, we need to invoke the chain with `chain.invoke`, providing a dictionary of key-values that get injected into `PromptTemplate`'s `input_variables`.  
  
Heres what the code looks like putting everything together, to get a summary about Steve Jobs:  
`ice_breaker.py`:  
```python  
from langchain_core.prompts import PromptTemplate  from langchain_openai import ChatOpenAI  from dotenv import load_dotenv    
    
information = """Steven Paul Jobs (February 24, 1955 – October 5, 2011) was an American businessman, inventor, and investor best known for co-founding the technology company Apple Inc. Jobs was also the founder of NeXT and chairman and majority shareholder of Pixar. He was a pioneer of the personal computer revolution of the 1970s and 1980s, along with his early business partner and fellow Apple co-founder Steve Wozniak.    
    
... a bunch more information pasted in from Wikipedia """    
    
if __name__ == "__main__":  
    load_dotenv()  
    summary_template = """        given the information {information} about a person, I want you to create:        1. a short summary        2. two interesting facts about them    """  
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)    
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")  
    chain = summary_prompt_template | llm    
    res = chain.invoke(input={"information": information})  
    print(res)  
```  
  
Testing our script, the output gives us the text output from the LLM, and a bunch of metadata:  
  
```  
content='1. Steven Paul Jobs was an American businessman, inventor, and investor best known for co-founding Apple Inc. He was a pioneer of the personal computer revolution and played a key role in the development of iconic products such as the Macintosh, iPod, iPhone, and iPad.\n\n2. Two interesting facts about Steve Jobs:\n- Jobs was a college dropout, having attended Reed College briefly before dropping out. Despite not completing his formal education, he went on to become one of the most influential figures in the technology industry.\n- Jobs was known for his intense focus on design and aesthetics, leading to the creation of sleek and user-friendly products that revolutionized the tech industry. His collaboration with designer Jony Ive resulted in iconic products like the iMac and iPhone.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 152, 'prompt_tokens': 663, 'total_tokens': 815, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-Bq7HJyK6ZuKaJQKbuawdO3W3uzEpj', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='run--1e100589-aaae-4910-b92d-80205c973789-0' usage_metadata={'input_tokens': 663, 'output_tokens': 152, 'total_tokens': 815, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}  
```  
  
The output here is kind of unwieldy, but we could have pretty easily actually gotten just the content field by adding a parser to our chain: `chain = summary_prompt_template | llm | StrOutputParser()`, just make sure you import it from `langchain_core.output.parsers`.  
  
> Running this first query, we should note that LangChain tracing is on by default, which means every prompt, model response, and metadata for a run is sent to the LangSmith cloud dashboard operated by the LangChain team. This lets you debug and see data, but also means moving data off machine which we may not want to do at this moment. This can be disabled by setting env var `LANGCHAIN_TRACING_V2=false`.  
  
## Bringing in an external API  
  
Suppose we want to scrape some LinkedIn profiles, and use that data in a chain with some prompts to LLMs. We're going to use a service called [scrapin.io](scrapin.io) to do this, they have a great API where you paste in a URL and get a ton of structured data output. Its a paid service put they offer 100 free API calls which we'll take advantage of for this project.  
  
`third_parties/linkedin.py`:  
```python  
import os  import requests  import pprint  from dotenv import load_dotenv    
    
load_dotenv()    
    
def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):    
    """Manually scrape information from LinkedIn profiles,    
    mock setup with GitHub Gist for testing only"""    
    if mock:    
        linkedin_profile_url = "https://gist.githubusercontent.com/Tyler-Liquornik/0fd5161c15205545470a357ccb56162d/raw/215e217fda558fc83931f48df56c324e4353a428/satya-nadella-scrapin.json"    
        response = requests.get(linkedin_profile_url,timeout=10)    
    else:    
        api_endpoint = "https://api.scrapin.io/enrichment/profile"    
        params = {"apikey": os.environ["SCRAPIN_API_KEY"], "linkedInUrl": linkedin_profile_url}    
        response = requests.get(api_endpoint, params=params, timeout=10)    
    
    person_data = response.json().get("person")    
    final_data = {    
        k: v    
        for k, v in person_data.items()    
        if v not in ([], "", None)    
    
        # We plan to send this data to an LLM, and every token costs $$$    
        # In this example, we don't care about certifications,   
        # we want to lighten our payload and keep it contextually relevant         
        and k not in ["certifications"]    
    }    
    
    return final_data    
    
# For testing, we can add an entry point to this script  
if __name__ == "__main__":    
    pprint.pprint(    
        scrape_linkedin_profile(    
            linkedin_profile_url="https://www.linkedin.com/in/satyanadella/"    
        )    
    )  
```  
  
> We use a core python API here `pprint`, which is a great tool that offers pretty-printing JSON with proper indenting for easier readability. Simply call `pprint.pprint(str)` to use it.  
  
We also setup a mock here, using **GitHub Gist** to hold our mock data. I got the mock data by making a call to our desired mock LinkedIn profile (which you can actually do through `scrapin.io`'s frontend to make it even easier), and then pasted it into a GitHub gist. Then GitHub's servers hold the data, and we can have tests do actual network requests, but not use our valuable API credits that in theory for real production use cases we'd be paying for.  
  
At this point, we can call `scrape_linkedin_profile` from our main file, and input it into our existing prompt setup to get LLM processed summaries of any LinkedIn profile, replacing the hardcoded sample data from earlier.  
  
`ice_breaker.py`:  
```python  
from langchain_core.prompts import PromptTemplate  
from langchain_openai import ChatOpenAI  
from dotenv import load_dotenv  
from third_parties.linkedin import scrape_linkedin_profile    
    
if __name__ == "__main__":    
    load_dotenv()    
    
    summary_template = """    
        given the information {information} about a person, I want you to create:          
        1. a short summary          
        2. two interesting facts about them"""    
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)    
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")    
    
    chain = summary_prompt_template | llm    
    
    linkedin_data = scrape_linkedin_profile("https://www.linkedin.com/in/tyler-liquornik/")    
    
    res = chain.invoke(input={"information": linkedin_data})    
    
    print(res)  
```

## Going Agentic

A common limitation of LLMs is their static training cut-off. To pull in fresh information you need an **agent** that can query live sources. LangChain agents do this through **tool calls**, for like a web-search API calls or a database queries. At each step while the agent is carrying out a task, its underlying LLM decides which tool to call next, receives the result, and repeats the cycle until it returns a final answer, in a **chain of thought** process.

We can use an agent here to improve our current workflow, where instead of passing in a LinkedIn URL, we have the agent so a web search for it using just the persons name, introducing another layer of automation.

Developing agents, we use LangChains `Tool` API. Among its parameters, `Tool` takes in:

- `name: str` a name to identify the tool, referenced by the LLM to identify it
- `description: str`, this description is very important to guides the model on when and how to call the tool
- `func: Callable[..., str]` (equivalent to `Function<Object[], String>` in Java), the function to run when the tool is called.

We're also going to use one of LangChains pre-built agent types, which for us will be a **ReAct** agent in this course, with the name coming from the relationship between agentic reasoning and acting. The ReAct algorithm is based on the idea of a 3 step iterative process:

1. **Think**: writes a short thought, describing what it needs next.
2. **Act**: issues an action that triggers a tool call
3. **Observe**: reads the tool’s output, then loops back to Thought.

To ReAct, we simply pull from LangChains's hub to get the widely used [ReAct prompt](# You can find it at: https://smith.langchain.com/hub/hwchase17/react) written by one of the LangChain co-founders, Harrison Chase. Here's what our agent looks like:

`agents/linkedin_lookup_agent.py`:
```python
import os  
from dotenv import load_dotenv  
from langchain_openai import ChatOpenAI  
from langchain_core.prompts import PromptTemplate  
from langchain_core.tools import Tool  
from langchain.agents import (create_react_agent, AgentExecutor)  
from langchain import hub  
  
def lookup(name: str) -> str:  
	load_dotenv()
	
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")  
  
    template = """given the full name of {name_of_person} I want you to get me a link to their LinkedIn profile page. You must obtain their profile home page, and not a post by them. Your answer should only have a URL, and nothing else"""
    prompt_template = PromptTemplate(input_variables=["name_of_person"], template=template)  
  
    tools = [  
        Tool(  
            name="Crawl Google for Linkedin profile page",  
            func="?", # Placeholder for now  
            description="useful for when you need to get a LinkedIn profile page URL",  
        )  
    ]  
  
    # This prompt comes from the Co-Founder of LangChain     
    react_prompt = hub.pull("hwchase17/react")  
  
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)  
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  
    result = agent_executor.invoke(  
        input={"input": prompt_template.format_prompt(name_of_person=name)}  
    )  
  
    return result["output"]  
  
if __name__ == "__main__":  
    linkedin_url = lookup("Satya Nadella")
```

Now we're gonna need another third party integration to give our LLM access to search the web. For this project, we'll use a service called [Tavily](https://www.tavily.com/), a search API optimized for LLMs & RAG, and write a small function to call their search API.

`tools/tools.py`:
```python
from langchain_community.tools.tavily_search import TavilySearchResults  
  
def get_profile_url_tavily(name: str) -> str:  
    """Searches for a LinkedIn Profile Page URL from a person's name"""  
  
    search = TavilySearchResults()  
    return search.run(name)
```

Then, if we replace `func=get_profile_url_tavily` in our original function, the agent has the ability to call on the search tool when it sees fit. Running `linkedin_lookup_agent`, there is something VERY IMPORTANT to note here. LangChain’s `react_single_input` parser looks for the chain of thought tags `Thought: ... Action: ... Action Input: ... Observation` verbatim. When it can’t find them, it raises an OutputParserException, which crashes the chain. It's up to the LLM ultimately to produce these tags based on the ReAct system prompt we pulled from LangChain's hub, which means *weak models can create very brittle workflows*. In my testing for this, `gpt-4.1-nano` OFTEN CRASHED when I tried, and even when it didn't it could not properly follow instructions and basically never gave just a URL, and gave the whole tool call output. Replacing it with `gpt-4o-mini`, it ALMOST ALWAYS WORKED with correct output. This goes to show how choosing the right model and testing output is absolutely key. There are more options to play around with that allow retrying, and different forms of parsing, but ultimately model selection must be done with care.

Now, connecting everything back in our primary driver file with both Scrapin and Tavily, and doing some refactoring we have a working program to enter someones name and get some info about them (assuming they have a LinkedIn):

`ice_breaker.py`:
```python
from langchain_core.prompts import PromptTemplate  
from langchain_openai import ChatOpenAI  
from dotenv import load_dotenv  
from third_parties.linkedin import scrape_linkedin_profile  
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent  
  
def ice_break_with(name: str):  
    linkedin_url = linkedin_lookup_agent(name=name)  
    linkedin_profile = scrape_linkedin_profile(linkedin_profile_url=linkedin_url)  
  
    summary_template = """  
          given the information {information} about a person, I want you to create:          1. a short summary          2. two interesting facts about them      """  
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)  
    llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")  
  
    chain = summary_prompt_template | llm  
    res = chain.invoke(input={"information": linkedin_profile})  
    print(res)  
  
  
if __name__ == "__main__":  
    load_dotenv()  
    person_name = input("Break the Ice With: ")  
    print(ice_break_with(name=person_name))
```

