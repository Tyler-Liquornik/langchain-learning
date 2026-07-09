
## Learning Goal  

Use the Deep Agents framework as a batteries included, opinionated framework to build effective custom agent harnesses without spending too much manual effort on making design decisions that have already been implemented probably better than I could by previous AI engineers which come up commonly in agent harness design.

> This learning is up to date for July 2026. There may be code from LangChain outside the deep agents SDK specifically that looks different than than older [LangChain Notes](./langchain) since LangChain as a whole has been updated/improved significantly over time, especially as a newer / less stable framework.

## What is Deep Agents?

Deep agents in a customizable agent harness that's purpose built for complex real-world tasks. 

Formally, an agent's **harness** is the structure of systems in which an agent operates that allows it to effectively complete its tasks. This can be broken down into four categories:
- Execution environment: can include a filesystem, sandboxes, and interpreters to execute code
- Context management: get the model the right context at the right time for the given task
- Delegation: tools to help plan long range task and delegate work to subagents
- Steering: prompts, and keeping a human in the loop to gate critical actions the agent takes

Ultimately, the job of a harness is to get the model the right context at the right time for the given task.

## Building a Deep Agent

Let's start with a simple agent thats just an LLM API call. It's just a deep agent by name since it's using the `create_deep_agent` function, but is the simplest possible form of a deep like like the old LangChain `create_react_agent` or newer `create_agent` we are well familiar with.

`m1/m1.2_scratch_agent.py`:
```python
from deepagents import create_deep_agent

from models import model

agent = create_deep_agent(model=model)

result = agent.invoke({"messages": [{"role": "user", "content": "What is an LLM?"}]})

print(result["messages"][-1].content)
```

`models.py`:
```python
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-haiku-4-5")

# Will use this later
strong_model = init_chat_model("anthropic:claude-sonnet-4-6")
```

Running this with `uv run ./m1/m1.2_scratch_agent.py` yields the following output using Claude Haiku 4.5:

```
An LLM (Large Language Model) is a type of artificial intelligence system trained on massive amounts of text data to understand and generate human language.

**Key characteristics:**

- **Scale** — Trained on billions or trillions of tokens (words/subwords) from diverse text sources
- **Architecture** — Typically uses transformer neural networks with attention mechanisms to process language
- **Capabilities** — Can perform tasks like text generation, summarization, translation, question-answering, reasoning, and code generation without explicit task-specific training
- **Few-shot learning** — Often performs well on new tasks with just a few examples or instructions

**How they work:**

LLMs predict the next word (or token) in a sequence based on context. During training, they learn patterns, facts, reasoning abilities, and linguistic structure. At inference time, they generate text by repeatedly predicting the most likely next token.

**Examples:**

- GPT-4, GPT-3.5 (OpenAI)
- Claude (Anthropic) — which is what I am
- Llama (Meta)
- Gemini (Google)
- Mistral, Qwen, and others

**Limitations:**

- Can hallucinate or generate false information with confidence
- No real-time information (training data has a cutoff date)
- Prone to biases present in training data
- Can't truly "understand" — they're pattern-matching systems
- Computationally expensive to train and run

LLMs have become foundational to modern AI applications, powering chatbots, code assistants, content generation tools, and more.
```

> Running `uv run ./m1/m1.2_scratch_agent.py` executes that file directly as the program entry point, so inside the file Python sets `__name__ == "__main__"`  implicitly and runs the code immediately. This is similar to Java executing the class that contains `public static void main(String[] args)`, except Python can run a file by path without requiring it to be addressed as part of a package. In this mode, Python starts import lookup near the script location, but in your project it can also see the broader `python/` folder, which is why `from models import model` resolves to the top-level `python/models.py`. Running with `python -m ...` is closer to Java’s fully qualified package execution, like `java com.example.Main`: Python treats the file as a module inside a package and expects imports to line up more deliberately with that package structure. 

> Confusingly coming from Java, recall that in everyday Python, a **module** is (almost) always one `.py` file, while a folder containing multiple modules is a **package**. Familiarly in Java 9+ a module/project has packages which have files/classes/interfaces, equivalently in python a project has packages which have modules/files which may have classes in them. This means packages are analogous in both languages, but in Java modules are a level above packages while in python a level below.


#### System Prompt

Your system prompt goes into a deep agent like:

`m1/m1.4_scractch_agent_bulter.py (Sinppet)`:
```python
agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    name="Butler Agent",
)
```

#### Tools

Some useful built in tools that are built in to the deep agents SDK:

| Tool(s)                                      | What it does                                          |
| -------------------------------------------- | ----------------------------------------------------- |
| `ls`, `read_file`, `write_file`, `edit_file` | Read and write files in the agent's working directory |
| `glob`, `grep`                               | Search files by name pattern or content               |
| `execute`                                    | Run shell commands (requires a sandbox backend)       |
| `task`                                       | Delegate work to a sub-agent with an isolated context |
| `write_todos`                                | Manage a running to-do list for multi-step planning   |

You can add tools to a deep agent like:

```python
agent = create_deep_agent(
    model=model,
    tools=[my_tool_a, my_tool_b],   # merged with the built-in tools (ls, grep, task...)
)
```

> Recall tools like `my_tools_a` are just functions decorated with `@tool`. The current way to do this is to use `langchain_core.tools.tool` to get the decorator.

#### MCP

From prior [LangChain](./langchain) lessons, we are familar with MCP servers and recall specifically as an example LangChain MCP server itself for giving coding agents up to date documentation on how to write LangChain; an incredibly useful tool since LangChain frameworks are new and update frequently. 

We can connect to MCP servers using an MCP Client object that will give us a list of tools to pass directly into the agent with clean syntax.

`m1/m1.6_agent_mcp.py (Snippet)`:
```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from models import model

client = MultiServerMCPClient({
    "docs-langchain": {
        "transport": "http",
        "url": "https://docs.langchain.com/mcp",
    }
})
tools = await client.get_tools()

agent = create_deep_agent(model=model, tools=tools)
```

We can also setup authorization with OAuth like

```python
client = MultiServerMCPClient({
    "github": {
        "transport": "streamable_http",
        "url": MCP_URL,
        "auth": auth_provider,
    }
})
```

Where `auth_provider` is an instance of `mcp.shared.auth.OAuthClientProvider`. Note that `mcp` here is not explicitly declared in `pyporject.toml`, our manifest file, since `mcp` is a transitive dependency of `langchain-mcp-adapters`.

#### Messages, Threads, and Checkpointers

Deep agents make it easy to keep track of more than one model call. We categorize multi-turn behaviour into three runtime ideas: messages, threads, and checkpointers.

**Messages** look like:

`m1/m1.7_messages_threads_checkpointers.py (Snippet)`:
```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is an LLM?"}]
})
```

The main three types of messages that come up with DeepAgents are:

| Message          | Where it comes from                                              |
| ---------------- | ---------------------------------------------------------------- |
| **HumanMessage** | The human's input, such as `{"role": "user", "content": "..."}`. |
| **AIMessage**    | The model's response. It can contain text or `tool_calls`.       |
| **ToolMessage**  | The result returned after the Tool Node runs a tool.             |
There are of course other messages like SystemMessage too, and other more exotic ones.

**Threads** are an ongoing conversation or run history. If you want an agent to remember previous turns across separate `invoke()` calls, give those calls the same `thread_id`.

```python
config = {"configurable": {"thread_id": "my-thread"}}

agent.invoke(
    {"messages": [{"role": "user", "content": "Remember my favorite color is blue."}]},
    config=config,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "What is my favorite color?"}]},
    config=config,
)
```

**Checkpointers** save thread state between calls. Without a checkpointer, there is nowhere to store the history for later. For local dev, the simplest checkpoint is `MemorySaver()` which stores checkpoints in the running Python process. Proper persistent infrastructure is needed for threads to survive process restarts.

Checkpointers are what make multi-turn state reliable because they save the current thread, and the next call with the same `thread_id` to continue from that saved state

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_deep_agent(
    model=model,
    checkpointer=MemorySaver(),
)
```

#### Human-in-the-Loop

**Human-in-the-Loop (HITL)** is the idea that we should pause before a sensitive agent action and let a person approve, edit, or reject it. The main purpose for this is safety, to ensure the user is aware of potentially dangerous actions like sending an email, running a database query, deleting a file, etc. HITL lets the agent suspend exection while it waits instead of staying active indefinitely. 

There are four pieces to remember:

|Piece|What it does|
|---|---|
|`interrupt_on`|Names the tools that require human review before execution.|
|`checkpointer`|Saves the paused agent state. For local development, use `MemorySaver()`.|
|`thread_id`|Identifies the saved run state. Resume with the same `thread_id`.|
|`Command(resume=...)`|Continues the paused run with the human decision.|

