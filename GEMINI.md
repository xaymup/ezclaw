# EzClaw Project Instructions

## Architecture Rules

### Intent Analysis & Memory Recall
- Memory recall is managed through a "Search-First Analysis Protocol" in `agent.py`.
- Instead of hardcoded keywords, the agent uses a self-prompting mechanism to analyze if a request requires context from long-term memory.
- **Priority**: Trigger memory recall if the user mentions personal details (names, locations), past projects, specific tasks implying history, or preferences.
- **Precision (Signal-over-Noise)**:
    - `search_memories` uses relevance filtering: queries with 3+ keywords require at least 2 matches.
    - Recall results are limited to the top 5 most relevant facts.
    - Background context baseline (`get_key_facts`) is strictly limited to 8 high-value or recent facts.
- **Self-Correction**: The intent analyzer includes a retry loop to handle JSON parsing failures, ensuring robust intent detection without "dumb" fallbacks.


## Tool Integration
- Tools are registered in `tools.py` using the `@registry.register` decorator.
- `ChatAgent` uses `registry.get_tool_definitions()` to obtain Ollama-compatible JSON schemas for all registered tools.
- Tool execution is handled in the `chat_stream` loop by looking up the function name in `registry.tools`.

### Multi-Agent Routing (Super-Architect)
- The project uses an iterative **Super-Architect** pattern for complex tasks.
- **Orchestration**: The Architect (`phi4-reasoning:plus`) creates a multi-step plan and delegates sub-tasks sequentially.
- **Loop Detection**: The system uses a `(agent_key, plan)` hashing mechanism to detect and stop redundant loops (e.g., if the Architect keeps sending the same plan to the same agent).
- **Context Management**: 
    - The Architect maintains a `task_context` that summarizes older steps while keeping full details for the most recent 3 steps.
    - Error detection is nuanced: it only triggers the `debugger` for active failures, not for mentions of resolved errors.
- **Workspace Awareness**: Agents follow a "Path Verification Protocol," checking both root and `workspace/` directories and verifying file content before taking action to avoid redundant copies.
- **Interactive Shell Handling**: 
    - For commands requiring user input (e.g., `sudo`, `pacman`, `vim`), agents must use `interactive=True` in `run_shell`.
    - This triggers a TUI suspension, allowing the user to interact directly with the terminal.
    - Agents are instructed to inform the user when an interactive session is starting.
- **Iterative Review**: After an agent completes a task, the Architect reviews the result and decides if the next step is needed.
- **Workflow**:
    1. Architect analyzes the request and provides a `plan`.
    2. Architect delegates the next step with specific `reasoning`.
    3. Specialized Agent executes and returns output.
    4. Architect reviews output and repeats or finalizes.

