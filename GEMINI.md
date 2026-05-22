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
- **Iterative Review**: After an agent completes a task (e.g., Researcher finds information), the Architect reviews the result and decides if the next step (e.g., Executor writing code) is needed.
- **Workflow**:
    1. Architect analyzes the request and provides a `plan`.
    2. Architect delegates the next step with specific `reasoning`.
    3. Specialized Agent executes and returns output.
    4. Architect reviews output and repeats or finalizes.
- **Context Management**: The Super-Architect maintains a `task_context` that accumulates results across iterations, ensuring consistency during multi-model handoffs.

