> Disclaimer: The content here comes from LangChain Academy's Deep Agents introductory course. The majority of if is copied directly from the course, with some of my thoughts and notes sprinkled throughout, and paraphrasing and choice omission of passages at times too.
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

## The Execution Environment

A deep agent runs inside an environment where it can store files, run commands, and optionally execute code inside the agent loop.

A Deep Agent environment comes with:
- A filesystem for reading and writing files.
- A shell, optionally. Exposed through the `execute` tool when the environment supports command execution.
- An interpreter, optionally. A separate capability for running code inside the agent loop.

###### File System

A filesystem gives the agent a place to work. It can use files as a scratchpad for notes, intermediate results, plans, and generated artifacts.

Deep Agents always provide filesystem tools such as `ls`, `read_file`, `write_file`, `edit_file`, `glob`, and `grep`. The agent sees the same tool surface even if the backend implementation changes.

That backend detail matters when you care where files live, but this lesson only needs the core idea: the filesystem is always part of the environment.

###### Shell

A shell lets the agent run commands through the `execute` tool: scripts, tests, package commands, and other process-level work. This is powerful, so the implementation matters.

Different backends can provide different implementations of `execute`. Two common families are:

**Sandbox providers** run commands in isolated environments such as containers, VMs, or remote sandboxes. This limits what code can affect compared with running directly on your machine, but safety still depends on sandbox configuration, network access, credentials, and provider isolation.

**Local shell** runs commands directly on your machine. It has no isolation overhead and can access local resources, but it should be used only when that local access is actually required.

The key distinction: `execute` does not inherently mean to run locally; it runs wherever the configured backend says commands should run.

###### Interpreter

An interpreter is different from a shell. It does not run through the backend. It is a separate tool/capability inside the agent loop.

Use an interpreter when the agent needs code-like control flow without starting a full shell process. For example, instead of asking the model to call a tool 100 times, the interpreter can run a loop, collect intermediate results in variables, and return only the final summary to the model.

Most agent work alternates between model reasoning and tool calls. A model can fire several tool calls in one turn, but that batch is fixed the moment it is emitted. Nothing can loop, branch on a result, retry a failure, or feed one call's output into the next without another model turn, and every result returns to the model's context.

Interpreters give the agent a runtime for that work. A loop runs every iteration, intermediate values stay in variables, and only a compact result returns to the model.

#### Filesystem Backends

Now that you have seen the environment surface, we can name the implementation layer.

The **environment** is what the agent sees: filesystem tools, optional `execute`, and optional interpreter. The **backend** is what implements the filesystem and optional shell behind that environment.

A backend answers questions like:
- where do files live?
- is `execute` available?
- if `execute` is available, where do commands run?

There is an abstract backend interface with multiple implementations. Some implementations only provide filesystem storage. Some also provide shell execution. Shell execution can be backed by a local shell or by different sandbox providers.

The key point is that the agent's tool surface can stay the same while the backend changes. The agent can still call `read_file` or `write_file`; the backend decides what those calls mean underneath.

The interpreter is not part of this backend layer. It is a separate capability added to the agent loop.

![filesystem-backend](./images/filesystem-backend.png)

You can select from different backends to implement:
- **StateBackend**: the default. Files live in the agent's saved state for the current thread. This is fast and zero-config, and it is good for scratch work.
- **FilesystemBackend / local disk**: reads and writes files on the local disk, scoped to a `root_dir` you specify. Use it when the agent should work with real files in a local directory.
- **CompositeBackend**: routes different path prefixes to different backends. It lets one agent use StateBackend for normal scratch files while routing a specific directory, such as `/reference/`, to local disk.
- **Custom backend**: implement the backend interface yourself to plug in another storage system, such as S3, GitHub, or a proprietary store.

Pass any backend directly to `create_deep_agent`. If you do not pass one, Deep Agents uses `StateBackend()` by default.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir="/path/to/project", virtual_mode=True),
)
```

Sometimes one backend is not enough. You may want ordinary scratch files to stay in thread state, but a specific directory to map to real local files.

`CompositeBackend` resolves this by routing path prefixes to different backends. Everything not matched by a route falls through to the `default` backend.

```python
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

reference_dir = Path(__file__).parent / "reference"

agent = create_deep_agent(
    model=model,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/reference/": FilesystemBackend(
                root_dir=str(reference_dir),
                virtual_mode=True,
            ),
        },
    ),
)
```

> `/reference/notes.md` goes to a local file at `reference/notes.md`, Everything else goes to `StateBackend`, scoped to the current thread's saved state

###### Permissions

Permissions are enforced in code, not in the prompt. The model cannot bypass them. Pass a list of `FilesystemPermission` rules to `create_deep_agent`; rules are evaluated in order and the first match wins. If nothing matches, the operation is allowed.

Each rule has three fields:

| Field        | Values                             | Default   | Description                                                                                    |
| ------------ | ---------------------------------- | --------- | ---------------------------------------------------------------------------------------------- |
| `operations` | `"read"`, `"write"`                | -         | `"read"` covers `ls`, `read_file`, `glob`, `grep`. `"write"` covers `write_file`, `edit_file`. |
| `paths`      | glob patterns                      | -         | e.g. `["/reference/**"]`. Supports `**` and `{a,b}` alternation.                               |
| `mode`       | `"allow"`, `"deny"`, `"interrupt"` | `"allow"` | `"interrupt"` pauses for human approval instead of blocking.                                   |
```python
from pathlib import Path

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

reference_dir = Path(__file__).parent / "reference"

agent = create_deep_agent(
    model=model,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/reference/": FilesystemBackend(
                root_dir=str(reference_dir),
                virtual_mode=True,
            ),
        },
    ),
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/reference/**"],
            mode="deny",
        ),
    ],
)
```

> The agent can read `/reference/`, but it cannot write to it. Other paths still use the default `StateBackend` route.

#### Sandbox and LocalShell Backends

The filesystem backends from the last lesson give the agent file tools: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`. `LocalShellBackend` and sandbox backends add one more: **`execute`**, which runs shell commands.

![shell-backend](./images/shell-backend.png)

Shell-capable backends expose an `execute(command)` tool. The agent calls it to run scripts it has written, invoke CLI tools, and compile and test code. The combined command output, exit code, and execution metadata come back as a tool result on the next LLM call.

###### How do you choose between LocalShell and Sandboxes?

Sandboxes are used for isolation. They are designed to let agents execute code, access files, and use the network away from your host machine. They are safer than running directly on your machine, but safety still depends on sandbox configuration, network access, credentials, and provider isolation.

In LangSmith Sandboxes, credentials are kept out of the sandbox entirely via an **auth proxy** pattern: the sandbox code makes API calls with no authentication headers, and a proxy sidecar intercepts outbound requests and injects the credentials on the way out. Secrets never enter the sandbox. The agent cannot exfiltrate what it cannot see.

Sandboxes are especially useful for:
- **Coding agents:** Agents that run autonomously can use shell, git, clone repositories (many providers offer native git APIs, e.g., Daytona's git operations), and run Docker-in-Docker for build and test pipelines
- **Data analysis agents:** Load files, install data analysis libraries (pandas, numpy, etc.), run statistical calculations, and create outputs like PowerPoint presentations in a safe, isolated environment

`LocalShellBackend` gives the agent access to the local filesystem and shell. Even with filesystem path scoping, the `execute` tool runs with host permissions; file scoping does not limit what shell commands can do. Unless your intention is to build a desktop agent designed to work on local files and commands, a sandbox is a better choice.

#### Sandbox Backends

A sandbox is a temporary, isolated workspace containing a filesystem, command execution environment, and other resources. Sandbox backends run commands inside that workspace instead of on your host machine.

There are two models for using a sandbox with an agent:

![sandbox-backend](./images/sandbox-backend.png)

#### LocalShell Backends

`LocalShellBackend` runs commands directly on the host machine. It can scope filesystem _tools_ to a `root_dir`, but `execute` itself runs with host permissions; no process isolation is applied.

Use `virtual_mode=True` when you want filesystem tools (`ls`, `read_file`, `write_file`, etc.) to be path-scoped under `root_dir`. Either way, `root_dir` does not restrict the `execute` tool. Shell commands run with host permissions and can access paths outside the filesystem-tool root. That's why LocalShellBackend is unsuitable for production or untrusted input.

This backend grants agents direct filesystem read/write access and unrestricted shell execution on your host. Use with extreme caution and only in appropriate environments.

**Appropriate use cases:**

- Local development CLIs (coding assistants, development tools)
- Personal development environments where you trust the agent's code
- CI/CD pipelines with proper secret management

**Inappropriate use cases:**

- Production environments (such as web servers, APIs, multi-tenant systems)
- Processing untrusted user input or executing untrusted code

**Note:** `virtual_mode=True` provides no security with shell access enabled, since commands can access any path on the system.

## Interpreters

**Interpreters are a lighter-weight code execution option compared to backends.** They embed a JavaScript runtime directly in the agent loop with no cloud infrastructure and no API calls to spin up a shell environment. The tradeoff is a narrower direct capability set: JavaScript standard library only, no external packages, no direct filesystem, and no direct network access.

When `CodeInterpreterMiddleware` is added to an agent, it provides a single new tool: **`eval`**. The agent calls it with a string of JavaScript to execute. The middleware runs the code in a [QuickJS](https://bellard.org/quickjs/) runtime, a lightweight JavaScript engine designed for embedded execution. The result of the last expression, plus any `console.log` output, comes back as the tool result.

QuickJS makes the following JavaScript function / packages avaialble:
- Array methods: `map`, `filter`, `reduce`, `sort`, `flat`, `find`  
- `Map`, `Set`, `JSON`, `Math`, `Promise`  
- `Date` and other standard JavaScript globals  
- `console.log` (captured and returned as output)  
- String and number methods, destructuring, `async`/`await`

QuickJS does not have available:
- Filesystem (`fs`)  
- Network (`fetch`)  
- Node.js APIs  
- `npm` packages

The interpreter state persists across `eval` calls within the same thread. Variables defined in one call are available in the next.

Interpreter execution is bounded by runtime limits such as timeout, memory, output size, and PTC call limits. Those limits keep runaway loops or huge results from taking over the agent run.

Good use cases for interpreters includes:
- Data transformation: When the agent has data in context and needs to compute something (sort, group, aggregate, reformat), `eval` is faster and more reliable than asking the LLM to do arithmetic in prose.
- Programmatic Tool Calling (PTC): JavaScript code can call the agent's tools directly, without those intermediate results ever entering the LLM's context window

###### Interpreter vs shell-capable backend

| Capability         | Interpreter                                                               | Shell-capable backend / sandbox                                    |
| ------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Language           | JavaScript (QuickJS)                                                      | Any shell command or installed runtime                             |
| External libraries | JS standard library directly; more only through allowlisted tools         | Packages available in the environment                              |
| Filesystem         | No direct filesystem API; can only access files through allowlisted tools | Filesystem tools plus shell access when supported                  |
| Network            | No direct network API; can only access network through allowlisted tools  | Depends on backend and sandbox/network configuration               |
| Infrastructure     | Embedded in the agent process                                             | Local shell or external sandbox resource                           |
| State              | Persists within thread                                                    | Persists within the shell/sandbox workspace until reset or deleted |
| Best for           | Computation, loops, PTC orchestration                                     | Scripts, packages, file I/O, builds, tests, arbitrary shell work   |

Use an interpreter when the task is logic and orchestration. Use a shell-capable backend or sandbox when the task requires packages, filesystem operations from code, network access, or arbitrary command execution.

## Context Management

There are many methods of doing context management that together prevent from overloading the a deep agent's context window while ensuring the right context is loaded at the right time for the given task.

#### Summarization

Long-running agents accumulate large message histories. Summarization works on messages in the context window and intelligently summarizes them. Offloading intercepts messages (human or tool messages) before they are inserted into context and saves them to the filesystem, inserting a pointer to the information in the context. These are detailed below.

When the context hits 85% (by default), the middleware picks a cutoff point that keeps the most recent messages intact and summarizes older messages into a single block. The messages being summarized are appended to /conversation_history/{thread_id}.md on the filesystem. The summary message the model receives includes a link to that file, so the agent can read those older sections back with read_file if it needs the detail. Because a summary is far more compact than the messages it replaces, the token count drops sharply and the conversation can continue. This is equivalent to `/compact` with Claude Code.

A summary containts:
- **Session intent**: what the user is trying to accomplish
- **Key facts & decisions**: important context from the conversation
- **Artifacts**: files or resources created or modified
- **Next steps**: what remains to do

#### Context Offloading

Summarization compacts messages that have already accumulated in context. Offloading handles a different case: a single large input. Instead of letting it land in context, **FilesystemMiddleware** intercepts oversized content before it reaches the model. It writes the full data to the filesystem and leaves behind a short preview plus a file path. The agent reads the full content with `read_file` only when it needs it.

Two kinds of data are offloaded this way: large **tool results** and large **human messages**.

###### Tool Call Offloading

A single tool call can return a large result, such as a full database export, hundreds of search matches, or a long code generation output. Storing it inline would use a large share of the context window at once.

**FilesystemMiddleware** handles this automatically with tool result offloading. When a tool returns a result that exceeds roughly **20,000 tokens**, the middleware:

1. Writes the full result to `/large_tool_results/<tool_call_id>` on the agent's filesystem
2. Replaces the inline `ToolMessage` with a short preview (first and last 5 lines) and a file reference

The preview looks something like:
```
Tool result too large, the result of this tool call <id> was saved in the
filesystem at this path: /large_tool_results/<tool_call_id>

You can read the result from the filesystem by using the read_file tool,
but make sure to only read part of the result at a time.

Here is a preview showing the head and tail of the result:

     1  | SELECT id, name, email FROM users WHERE ...
     2  | 1, Alice, alice@example.com
     3  | 2, Bob, bob@example.com
     4  | 3, Carol, carol@example.com
     5  | 4, Dave, dave@example.com
... [1 823 lines truncated] ...
  1828  | 1827, Zara, zara@example.com
  1829  | 1828, Zoe, zoe@example.com
  1830  | 1829, ...
```

The agent can then call `read_file` with `offset` and `limit` to page through the full result on demand - reading only the portion that is relevant rather than loading everything into context at once.

The tools `ls`, `glob`, `grep`, `read_file`, `edit_file`, and `write_file` are excluded from offloading. They either have their own built-in truncation, return minimal confirmation text, or (for `read_file`) re-reading a truncated file would not help.

###### Human Message Offloading

The same mechanism applies when a _user_ message is very large, for example pasting in a long document or transcript. When a human message exceeds roughly **50,000 tokens**, FilesystemMiddleware writes the full text to `/conversation_history/` and replaces it in the model's view with a preview and a file reference. The agent can `read_file` that path to retrieve the full message on demand.

The behavior the agent sees, a preview plus a pointer, is identical to a tool result. Only the threshold (50k vs 20k tokens) and the storage location differ.

#### Skills

A skill is a directory containing a `SKILL.md` file. Skills follow an [open standard](https://agentskills.io/specification) maintained by many players like OpenAI, Microsoft, Google, Cursor, and more. They are portable across agents and shareable across teams.

A skill directory looks like:
```
skill-name/
├── SKILL.md          # Required: metadata + instructions
├── scripts/          # Optional: executable code
├── references/       # Optional: documentation
├── assets/           # Optional: templates, resources
└── ...               # Any additional files or directories
```

The `SKILL.md` file has two parts: a YAML frontmatter block at the top, and the skill instructions below it. The agent reads only the frontmatter at startup; the full body is loaded when a skill is activated. Write the `description` to clearly answer when the skill should be used. Keep it brief and specific.

Frontmatter fields in the official skill spec is as follows:

|Field|Required|Notes|
|---|---|---|
|`name`|yes|Lowercase, alphanumeric, hyphens only. Must match the directory name.|
|`description`|yes|Describes when to use this skill. Kept brief; always in context.|
|`allowed-tools`|no|Space-separated list of tool names the skill may use.|
|`compatibility`|no|Environment or version requirements.|
|`metadata`|no|Arbitrary key-value pairs.|
###### Progressive Disclosure

Skills use a three-stage pattern to keep the agent's context lean:

**1. Discovery**

The SDK reads the `name` and `description` from each skill's frontmatter and automatically injects them into the system prompt on every call. You do not write this section; the SDK generates it. Simplified, it looks like this:

```sql
## Skills System

Available skills:
- **qualify-lead**: Use when the user wants to qualify a sales lead or prospect.
  -> Read skills/qualify-lead/SKILL.md for full instructions.
- **draft-pitch**: Use when the user wants to write a sales pitch or outreach message.
  -> Read skills/draft-pitch/SKILL.md for full instructions.
```

Both the `name` and `description` are always in context, so keep them short and precise. The `description` in particular should describe only **when** to use the skill, not what it does internally.

**2. Activation**

When a task matches a skill's description, the agent calls `read_file` on the path shown in the system prompt, loading the full `SKILL.md` into context for that turn. The full content is not loaded automatically; the agent reads it on demand.

**3. Execution**

The agent follows the instructions in the skill body, calling any tools it needs and using any bundled files referenced in the instructions.

You can see all three stages in a LangSmith trace: the Skills System section appears in every system message; the `read_file` call marks the moment of activation; and the subsequent tool calls show execution.



Once you author skills, put them into the agent'ts backend and register them:

```python
from deepagents.backends import FilesystemBackend

backend = FilesystemBackend(root_dir="/path/to/agent-files", virtual_mode=True)

agent = create_deep_agent(
    model=model,
    backend=backend,
    skills=["/skills"],
)
```

At runtime, the agent:
1. Sees all skill names and descriptions in the system prompt
2. Matches the user request to a skill description
3. Calls `read_file` on that skill's `SKILL.md`
4. Follows the full instructions in the skill body

#### Memory

Memory lets an agent carry durable knowledge across conversations: user preferences, project conventions, team workflows, recurring facts, and instructions the base model would not know on its own.

> Like a s
> 
> ystem prompt, memory files are always loaded into context

The key distinction is:

- **Checkpointers preserve thread history**: messages and state within a conversation thread.
- **Memory preserves durable knowledge**: file-backed facts and instructions that can be loaded into future runs, including different threads when the backend scope allows it.

In Deep Agents, memory is plain files on the agent filesystem. The common convention is a markdown file such as `/memories/AGENTS.md`. For example:

```markdown
# Project Guidelines

## Code Style
- Use type annotations
- Prefer pathlib.Path for file operations

## Workflow
- Run tests with: uv run pytest
```

The agent can update memory with the same filesystem tools it already knows, especially `edit_file`. There is no separate memory database API for the agent to learn.

The `memory` argument tells Deep Agents which files should be treated as memory:

```python
agent = create_deep_agent(
    model=model,
    memory=["/memories/AGENTS.md"],
)
```

Without `memory=[...]`, the file may still exist in a backend, but it is not loaded into the system prompt as agent memory.

Passing `memory=[...]` creates `MemoryMiddleware` behind the scenes.

For each run, the middleware:

1. Loads the configured memory files from the backend
2. Stores the loaded contents in agent state
3. Appends the combined memory to the system prompt inside an `<agent_memory>` block on each model call in that run

Here's a breakdown of Memory vs. Skills

|                              | Memory                         | Skills                                     |
| ---------------------------- | ------------------------------ | ------------------------------------------ |
| What is always visible?      | Loaded memory content          | Skill names and descriptions               |
| When is full content loaded? | Proactively for the run        | On demand when the agent activates a skill |
| Typical use                  | Durable facts and instructions | Reusable workflows and procedures          |

###### How memory is updated

The middleware also adds instructions telling the agent when to persist new information. When a user says "remember this" or provides context that should carry forward, the agent can call `edit_file` on one of the paths passed to `memory=[...]`.

That write updates the backend file. The new content is available after memory is reloaded in a later run.

A typical memory write looks like:

```text
User: Remember: the team switched to ruff for linting.
Agent: edit_file('/memories/AGENTS.md', ...)
```

Only store information that is safe and useful in future conversations. Do not store secrets such as API keys, credentials, tokens, or passwords.

###### Where memory lives

The backend controls where memory files are stored. A common production pattern is to use a `CompositeBackend`: keep normal working files in the default backend, and route a dedicated `/memories/` directory to durable storage.

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()


def memory_namespace(runtime):
    user_id = runtime.context["user_id"]
    workspace_id = runtime.context["workspace_id"]
    return ("memory", workspace_id, user_id)


agent = create_deep_agent(
    model=model,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(namespace=memory_namespace)},
    ),
    store=store,
    memory=["/memories/AGENTS.md"],
)
```

In this setup, `/memories/AGENTS.md` is the logical path the agent sees. The `CompositeBackend` routes that path into `StoreBackend`, so memory persists beyond a single thread. Other working files can still use the default backend.

`StateBackend` is thread-scoped, so it is useful for temporary working files but not for durable long-term memory. `InMemoryStore()` is only for local development; in production, the store is usually backed by durable infrastructure.

###### Scoping memory

The `StoreBackend` namespace on the `/memories/` route determines which durable memory files the agent sees. In a real application, you usually derive that namespace from runtime context such as the authenticated user, workspace, tenant, or assistant ID.

For example, a user-and-workspace scoped namespace might look like this:

```python
def memory_namespace(runtime):
    user_id = runtime.context["user_id"]
    workspace_id = runtime.context["workspace_id"]
    return ("memory", workspace_id, user_id)
```

Then invoke the agent with that context:

```python
agent.invoke(
    {"messages": [{"role": "user", "content": "Remember that I prefer concise answers."}]},
    context={"user_id": "u_123", "workspace_id": "acme"},
)
```

Common scoping patterns:

|Scope|Namespace shape|Use when...|
|---|---|---|
|Shared assistant memory|`("memory", assistant_id)`|Everyone should share the same durable instructions|
|User memory|`("memory", user_id)`|Preferences should follow one user across threads|
|Workspace memory|`("memory", workspace_id)`|A team/project should share conventions|
|User within workspace|`("memory", workspace_id, user_id)`|Users have private preferences inside a workspace|
|Assistant + user|`("memory", assistant_id, user_id)`|The same user may have different memories for different assistants|
Choose the scope deliberately. If private user memories share the same namespace, one user's preferences or context can leak into another user's agent run. If shared team memory is scoped too narrowly, the agent will fail to reuse conventions that should apply across the workspace.

## Delegation

Long-running agents need coordination. A big task can span many steps, tool calls, files, and intermediate results. Without an explicit way to track the plan and isolate specialized work, the main agent can drift or overload its context.

Use a subagent when:  
- A multi-step task would otherwise clutter the main agent's context  
- The work needs a specialized domain: custom instructions or its own tools  
- A subtask is better served by a different model  
- You want the main agent to stay focused on high-level coordination

Skip the subagent when:  
- The task is simple and single-step, so the overhead isn't worth it  
- You need to keep the intermediate context, not just the final result

###### Planning

A capable model can draft a solid plan for a multi-step job. The hard part is follow-through. As the conversation grows and the agent juggles tool calls and intermediate results, it tends to drift, losing track of what's already done and what's still left.

Deep Agents addresses this with a built-in **`write_todos`** tool. The agent writes its plan out as a structured to-do list and updates each item's status as it works:
- **`pending`**: not started yet
- **`in_progress`**: being worked on now
- **`completed`**: done

The list is saved in the agent's state. Across turns, it persists when you use the same thread with a checkpointer, following the thread pattern introduced earlier in the course. On a long task the agent always has an explicit, up-to-date plan to check against instead of trying to hold the whole thing in its head. Think of it as writing the steps on a whiteboard and ticking them off as it goes.

Planning is not delegation. It is the supervisor's task list: a way for the main agent to keep itself organized before, during, and after tool calls.

###### Delegation

When a project is too big for one person, we bring in a team. The same applies to agents.

A few things make this work:
- **A supervisor splits the work and pulls it back together.** They break the project into subtasks, hand each one out, and combine what comes back.
- **Each person brings their own expertise.** A researcher, a writer, and a designer each see the problem differently and carry their own tools.
- **People work in parallel.** Three people working at once finish in roughly a third of the time. BUT a common saying is more engineers is not always better, thats why its *roughly* a third, because of collaboration tax, which can be minimized IF the team manager is effective!
- **Each person focuses on their piece, not the whole project.** The writer doesn't need to hold the entire plan in their head, just their assignment.

We achieve delegation with **subagents**. A **subagent** is a full agent the main agent can call to do a focused task and report back. It can have its own instructions, tools, skills, model, and isolated context. The team pattern carries straight over.
- **The main agent splits work and aggregates results.** Say the job is _"write a newsletter summarizing this week's news on a topic."_ The main agent can hand each sub-topic to a different subagent. Each one searches, summarizes what it finds, and passes back only the summary, which the main agent stitches into the final newsletter.
- **Each subagent is a full agent in its own right.** It can have its own system prompt, its own skills, its own tools, and it can even run on a different model than the main agent.
- **Subagents run in parallel.** In Deep Agents, a subagent is invoked like a tool, so the main agent can fire off as many as it needs at the same time.
- **Each subagent focuses on its own task and its own context.** This is what makes subagents a context-engineering tool.

Deep Agents exposes subagents through a built-in `task` tool. To the main agent, delegating looks just like calling any other tool. The main agent calls `task` with an assignment and gets a result back, picking which subagent to use from each one's description, the same way it decides to call any other tool.

###### Context Isolation

**Context Isolation** is what makes subagents a context-engineering tool. A subagent does its work without cluttering the main agent's context:
- A subagent receives **one message** from the main agent describing its task. Importantly, it does **not** inherit the main agent's message history.
- It runs autonomously until the task is done, then returns **one message** back: its final result. LangChain Deep Agent subagents are stateless. They can't carry on a back-and-forth with the main agent.
- A subagent defines **its own tool set**. Consider a database expert with dozens of specialized tools. Putting those behind a subagent keeps every one of those tool descriptions _out_ of the main agent's context. All the main agent has to know is that this needs database expertise. It hands off the task and gets back an answer.

The payoff: a large, messy subtask gets compressed into a single clean result, as long as the subagent is told to return a concise summary. The main agent's context stays focused on coordination, even as the work piles up behind the scenes.

###### Synchronous vs. Asynchronous

Subagents can run in two modes. The choice comes down to one question: **does the main agent need the result before it can keep going?**.
- **Synchronous** subagents complete their task within the step. The main agent blocks until the longest-running subagent returns. It's the simplest model, but the conversation pauses while the work happens.
- **Asynchronous** subagents run in the background. The main agent launches the job, stays responsive, and checks on it later, collecting the result once it's ready. It's more responsive but has more moving parts.