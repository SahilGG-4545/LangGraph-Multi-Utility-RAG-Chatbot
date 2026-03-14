# 🤖 LangGraph Chatbot

A conversational AI chatbot built with **LangGraph**, **LangChain**, and **Streamlit**, powered by **Groq's ultra-fast inference** (Llama 3.3 70B). Supports multi-turn memory, multiple concurrent chat threads, real-time token streaming, and a polished animated UI.

![alt text](image.png)

---

## ✨ Features

- 🧠 **Stateful multi-turn conversations** — LangGraph `InMemorySaver` checkpointer persists message history per thread
- ⚡ **Real-time streaming** — tokens stream word-by-word via `chatbot.stream()`
- 💬 **Multiple chat threads** — create new conversations and switch between them freely from the sidebar
- 🎨 **Animated UI** — gradient title, fade-slide-in bubbles, bouncing typing indicator, floating empty-state, shimmer buttons
- 🔒 **Secure API key handling** — key loaded from `.env` via `python-dotenv`, never hard-coded

---

## 🗂️ Project Structure

```
chatbot/
├── langgraph_backend.py          # LangGraph graph definition, LLM config, compiled chatbot
├── streamlit_frontend_threading.py  # Main Streamlit UI (multi-thread, streaming)
├── main.py                       # CLI entry point (placeholder)
├── requirements.txt              # Pip-installable dependencies
├── pyproject.toml                # Project metadata (uv / PEP 517)
├── .env                          # 🔑 API keys (not committed)
├── extras/
│   ├── streamlit_frontend.py           # Basic single-thread frontend
│   ├── streamlit_frontend_streaming.py # Streaming-only frontend variant
│   └── langgraph_tool_backend.py       # Backend variant with tool-calling support
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.12+**
- A free [Groq API key](https://console.groq.com/)

### 1. Clone the repository

```bash
git clone https://github.com/SahilGG-4545/chatbot.git
cd chatbot
```

### 2. Create and activate a virtual environment

```bash
# Using uv (recommended)
uv venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# Or using plain venv
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
# With uv
uv add -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app

```bash
streamlit run streamlit_frontend_threading.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🏗️ Architecture

```
User Input
    │
    ▼
Streamlit Frontend   ←──── session_state (message_history, thread_id, chat_threads)
    │
    │  chatbot.stream()
    ▼
LangGraph Graph
  ┌─────────────────────────────────┐
  │  START → chat_node → END        │
  │                                 │
  │  chat_node:                     │
  │    llm.invoke(messages)         │
  │    (Groq / Llama 3.3 70B)       │
  └─────────────────────────────────┘
    │
    │  InMemorySaver (per thread_id)
    ▼
  Streamed AIMessage tokens → st.write_stream()
```

**Key design choices:**
- Each conversation is isolated by a UUID `thread_id` passed as LangGraph config
- `stream_mode="messages"` yields individual `AIMessage` chunks for low-latency streaming
- `chatbot.get_state()` is used to reload a thread's history when switching conversations

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM Inference | [Groq](https://groq.com/) — Llama 3.3 70B Versatile |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM Abstraction | [LangChain](https://github.com/langchain-ai/langchain) |
| Frontend | [Streamlit](https://streamlit.io/) |
| Env Management | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| Packaging | [uv](https://github.com/astral-sh/uv) / pip |

---

## 📦 Dependencies

```
langgraph
langchain-core
langchain-openai
python-dotenv
streamlit
```

---

## 🔐 Security

- `.env` is listed in `.gitignore` — **never commit your API key**
- The Groq key is read at startup and never exposed to the frontend

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push and open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
