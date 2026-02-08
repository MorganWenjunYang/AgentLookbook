# Agent Lookbook - Project Guidelines

## Project Philosophy

- **Zero framework dependency**: only `requests` for HTTP calls, no LangChain / LlamaIndex / OpenAI SDK / Anthropic SDK
- Each agent paradigm is a **self-contained subclass** of `BaseAgent`
- All LLM interaction goes through **raw API URLs** via `requests.post()`
- Designed for **learning and comparison** -- see how different reasoning paradigms behave on the same query

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)              │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐         │
│  │  Column 1  │ │  Column 2  │ │  Column 3  │  ...   │
│  │  (CoT)     │ │  (ReAct)   │ │  (Vanilla) │        │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘         │
└────────┼──────────────┼──────────────┼──────────────┘
         │              │              │
         ▼              ▼              ▼
┌─────────────────────────────────────────────────────┐
│              Agent Layer (agents/)                    │
│  BaseAgent.run(query) -> AgentResult                 │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │Vanilla │ │  CoT   │ │ ReAct  │ │  ...   │       │
│  └────┬───┘ └────┬───┘ └───┬────┘ └────┬───┘       │
└───────┼──────────┼─────────┼───────────┼────────────┘
        │          │         │           │
        ▼          ▼         ▼           ▼
┌──────────────────────┐  ┌──────────────────────────┐
│  LLM Layer (llm/)    │  │  Tool Layer (tools/)      │
│  LLMClient.chat()    │  │  Tool.run(input) -> str   │
│  ┌─────┐┌────┐┌────┐│  │  ┌──────────┐┌──────────┐│
│  │Qwen ││GLM ││DSek││  │  │Calculator││WikiSearch ││
│  └─────┘└────┘└────┘│  │  └──────────┘└──────────┘│
└──────────┬───────────┘  └──────────────────────────┘
           │
           ▼
   requests.post(api_url)
```

### Layer Responsibilities

| Layer | Module | Responsibility |
|-------|--------|---------------|
| **LLM** | `llm/` | Wrap raw HTTP calls to LLM APIs behind a unified `chat()` interface |
| **Tool** | `tools/` | Define tools with `name`, `description`, `run()`. Registry for prompt injection |
| **Agent** | `agents/` | Implement paradigm-specific reasoning loops. Produce `AgentResult` with trace |
| **UI** | `app.py` | Side-by-side comparison of paradigms on the same query |

## Paradigm Reference

### Implemented

| Paradigm | Core Idea | Paper | GitHub |
|----------|-----------|-------|--------|
| **Vanilla** | Direct LLM call, no special prompting. Baseline. | N/A | N/A |
| **CoT** | "Think step by step" -- elicit intermediate reasoning before the final answer | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903) | _TODO: add link_ |
| **ReAct** | Interleave Thought / Action / Observation in a loop until solved | [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023)](https://arxiv.org/abs/2210.03629) | _TODO: add link_ |
| **CodeAct** | Generate executable Python code as actions; self-debug via interpreter feedback | [Executable Code Actions Elicit Better LLM Agents (Wang et al., 2024 -- ICML)](https://arxiv.org/abs/2402.01030) | [xingyaoww/code-act](https://github.com/xingyaoww/code-act) |
| **Reflexion** | Self-reflect on failures and retry with accumulated verbal insight | [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023 -- NeurIPS)](https://arxiv.org/abs/2303.11366) | [noahshinn/reflexion](https://github.com/noahshinn/reflexion) |
| **InterCode** | Interactive coding: write code, observe execution, iterate until correct | [InterCode: Standardizing and Benchmarking Interactive Coding (Yang et al., 2023 -- NeurIPS D&B)](https://arxiv.org/abs/2306.14898) | [princeton-nlp/intercode](https://github.com/princeton-nlp/intercode) |

### Planned

| Paradigm | Core Idea | Paper | GitHub |
|----------|-----------|-------|--------|
| **ToT** | Branch into multiple reasoning paths, evaluate and select the best | [Tree of Thoughts (Yao et al., 2023)](https://arxiv.org/abs/2305.10601) | _TODO: add link_ |
| **ReWOO** | Plan all steps upfront, execute tools, then synthesize | [ReWOO (Xu et al., 2023)](https://arxiv.org/abs/2305.18323) | _TODO: add link_ |
| **BOLAA** | Controller dispatches to specialist sub-agents | [BOLAA (Liu et al., 2023)](https://arxiv.org/abs/2308.05960) | _TODO: add link_ |

## How to Add a New Paradigm

1. Create `agents/your_paradigm.py`
2. Subclass `BaseAgent`, set class attributes:
   ```python
   @register_agent
   class YourAgent(BaseAgent):
       paradigm_name = "YourParadigm"
       paradigm_description = "One-line explanation"
   ```
3. Implement `run(self, query: str) -> AgentResult`
   - Use `self.llm.chat(messages)` to call the LLM
   - Use `self.tools.get("tool_name").run(input)` for tool calls
   - Record each intermediate step as a `ThinkStep`
   - Return `AgentResult(answer=..., steps=[...], token_usage={...})`
4. Import your module in `agents/__init__.py`
5. It auto-appears in the Streamlit UI via the agent registry

## Coding Conventions

- **Python 3.11+**, type hints everywhere
- **dataclasses** for data structures (no Pydantic)
- **f-strings** for prompt templates, kept in the same file as the agent
- Each agent file is **self-contained** (prompts + logic together)
- Intermediate steps logged as `ThinkStep` for UI visualization
- No global mutable state; agents are instantiated per-request


## Paradigm Implement Guide

Go check the paper/official/unofficial implementation/prompt before you move, to find paper you can see pdf under ./paper/

- CoT: https://github.com/NirDiamant/Prompt_Engineering/blob/main/all_prompt_engineering_techniques/cot-prompting.ipynb
- GoT: https://github.com/PeterGriffinJin/Graph-CoT
- AdaPlanner: https://github.com/haotiansun14/AdaPlanner
- Intercode: https://github.com/princeton-nlp/intercode
- ReAct: https://github.com/langchain-ai/react-agent
- CodeAct: https://github.com/xingyaoww/code-act; https://github.com/langchain-ai/langgraph-codeact
- ToT: https://github.com/princeton-nlp/tree-of-thought-llm
- Reflexion: https://github.com/noahshinn/reflexion
- ADaPT: https://github.com/archiki/ADaPT
- ReCode: https://github.com/FoundationAgents/ReCode




## LLM Providers

All three providers use the OpenAI-compatible `/v1/chat/completions` endpoint:

| Provider | Base URL | Default Model |
|----------|----------|---------------|
| Qwen (DashScope) | `https://dashscope.aliyuncs.com/compatible-mode` | `qwen-plus` |
| GLM (智谱) | `https://open.bigmodel.cn/api/paas` | `glm-4-flash` |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |

## Current Scope

- **Phase 1** (current): Single-turn conversation, Vanilla + CoT + ReAct
- **Phase 2** (planned): Multi-turn conversation with memory retention
- **Phase 3** (planned): Real tool integration, more paradigms
