# EzClaw Ollama CLI

A powerful terminal-based assistant for Ollama with built-in tools and persistent memory.

## Why EzClaw?

EzClaw was created to be a minimal, focused terminal agent that "just works." Many other agents are either too complicated to configure, clutter the workspace too quickly, or have unreliable memory systems. 

EzClaw solves this by:
- **Zero-Fuss Configuration**: Get up and running in seconds.
- **Minimalist Design**: Focused strictly on the features you actually need.
- **Native Memory**: Built-in SQLite-backed memory that persists facts and history out of the box.
- **Clean Workspace**: Keeps operations organized and avoids cluttering your project root.

## Features
- **Persistent Chat History**: All conversations are saved in an SQLite database.
- **Long-term Memory**: The agent can explicitly `remember` facts and `recall` them later across different sessions.
- **Hybrid RAG Memory**: Uses a combination of full-text search and vector embeddings (Cosine Similarity) for high-precision memory recall.
- **Multi-Agent Orchestration**: Iterative **Super-Architect** pattern that decomposes complex tasks and routes them to specialized agents (Executor, Researcher, Debugger).
- **Tool Support**: 
    - `run_shell`: Execute terminal commands (interactive & background).
    - `read_file` / `write_file`: Manipulate files with unified diff previews.
    - `list_dir`: Explore the file system.
    - `web_fetch`: Search and synthesize information from the web.
    - `remember` / `recall`: Manage long-term state.
    - `learn_skill`: Save and reuse complex procedures and workflows.
- **Beautiful TUI**: Full-screen terminal interface with real-time thinking visualization, interactive tool authorization, and prompt history.

## Performance Metrics

EzClaw is optimized for high-performance local execution. Below are the benchmarks from our primary development machine:

**Hardware Profile:**
- **GPU**: NVIDIA GeForce RTX 4080 (16GB VRAM)
- **VRAM Utilization**: ~8.9 GB / 16.4 GB (running 14B models)
- **GPU Load**: ~8% during active inference
- **Inference Speed**: optimized for `qwen2.5-coder:14b` and `deepseek-r1:14b`

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Ollama**:
   Ensure Ollama is running and you have pulled a model that supports tool calling (e.g., `qwen2.5-coder:14b` or `llama3.1`).
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Update the `.env` file with your model name and hardware settings.

## Usage

Run the CLI:
```bash
python cli.py
```

### Example Commands
- "What files are in this directory?"
- "Create a new python file named hello.py that prints 'Hello World'."
- "Remember that my name is Alice and I like coffee."
- (In a new session) "Who am I and what do I like?"
- "Run `ls -la` and tell me the permissions of cli.py."

## Project Structure
- `cli.py`: The terminal user interface.
- `agent.py`: Core logic for Ollama communication and tool execution loop.
- `memory.py`: SQLite database management for history and facts.
- `tools.py`: Tool registry and implementation of default tools.
- `ezclaw.db`: SQLite database file (created on first run).
