
## Course Goal

The goal for this course is to be able to build AI applications in two domains: AI Agents, and Retrieval Augmented Generation. We would like to gain an understanding of the mechanisms that drive these applications under the hood, which means also being adept at Prompt Engineering, and understanding the LangChain Ecosystem stack, like using LangSmith for tracing, and LangGraph for workflow engineering. In addition to all this, we'll cover how to create production ready LangChain apps, with enterprise concerns like testing, logging & monitoring, security, and more.

Theres also gonna be some python basics in here, as I'm relatively new to building full fledged python applications that aren't just small scripts, and am writing from the perspective of a developer with more experience writing Java and JavaScript applications.

> This guide is updated for LangChain 0.3

## What is LangChain?

LangChain is an open source framework that simplifies the process of building LLM applications, with a set of useful abstractions that make it so we don't need understand the inner workings of LLMs, but rather use them as a blackbox for software engineering.

One example of this is that it lets us switch between different LLMs extremely easily, using the same interface for every model. This helps very easily prevent vendor lock with object design around one specific model.

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import Chat Ollama

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
from os

if __name__ == "__main__":
	 # Call this once at the start
	load_dotenv()
	# API Key value accessible assuming its in .env
	print(os.environ['API-KEY'])

```

As a bonus package, `pipenv install black`, for a linter, which you can just run right away with `black .` to keep our code neat.

A lot of the reason for all these different packages is the refactoring  through diverse contributions over time since LangChain started. For example before `langchain-openai`, you had to use the whole OpenAI SDK separately, but now its more unified with official integration packages that each LLM vendor maintains directly with LangChain.