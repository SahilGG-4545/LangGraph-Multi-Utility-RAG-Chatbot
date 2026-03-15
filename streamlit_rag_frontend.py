import html
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# ---- Page config (must be the very first Streamlit call) ----
st.set_page_config(
    page_title="LangGraph PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Global CSS + Animations ----
st.markdown(
    """
<style>
/* ═══════════════════════════════════════════════════
   KEYFRAME ANIMATIONS
═══════════════════════════════════════════════════ */

/* Gradient title hue shift */
@keyframes gradientShift {
    0%   { background-position: 0% 50%;   }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%;   }
}

/* Message bubble fade + slide up */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0);    }
}

/* Floating bob for empty-state icon */
@keyframes floatBob {
    0%, 100% { transform: translateY(0px);   }
    50%       { transform: translateY(-12px); }
}

/* Shimmer sweep for New Chat button */
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}

/* Sidebar active-thread glow pulse */
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 0px rgba(130,100,255,0.0); }
    50%       { box-shadow: 0 0 8px rgba(130,100,255,0.7); }
}

/* Typing dots bounce */
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0);    opacity: 0.4; }
    40%           { transform: translateY(-6px); opacity: 1;   }
}

/* Radial background pulse on main canvas */
@keyframes bgPulse {
    0%, 100% { opacity: 0.03; }
    50%       { opacity: 0.07; }
}

/* Tool status pulse */
@keyframes statusPulse {
    0%   { box-shadow: 0 0 0 0 rgba(96,165,250,0.45); }
    70%  { box-shadow: 0 0 0 8px rgba(96,165,250,0.0); }
    100% { box-shadow: 0 0 0 0 rgba(96,165,250,0.0); }
}

/* Tool status reveal */
@keyframes statusAppear {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ═══════════════════════════════════════════════════
   GLOBAL / MAIN AREA
═══════════════════════════════════════════════════ */

/* Subtle animated radial orb behind chat area */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: -30%;
    right: -20%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(130,100,255,1) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    animation: bgPulse 6s ease-in-out infinite;
    z-index: 0;
}

/* ═══════════════════════════════════════════════════
   ANIMATED GRADIENT TITLE
═══════════════════════════════════════════════════ */
.gradient-title {
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(270deg, #a78bfa, #60a5fa, #34d399, #f472b6, #a78bfa);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 5s ease infinite;
    margin-bottom: 0.1rem;
}

/* ═══════════════════════════════════════════════════
   CHAT MESSAGES — fade + slide in
═══════════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    animation: fadeSlideUp 0.35s ease both;
    border-radius: 12px;
    padding: 0.25rem 0.5rem;
    margin-bottom: 0.35rem;
}

/* ═══════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    padding-top: 1.5rem;
    border-right: 1px solid rgba(130,100,255,0.15);
}

/* All sidebar conversation buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    border-radius: 8px;
    padding: 0.45rem 0.75rem;
    font-size: 0.85rem;
    margin-bottom: 4px;
    border: 1px solid rgba(120,120,180,0.25);
    background: rgba(120,120,180,0.08);
    transition: background 0.2s, border-color 0.2s, transform 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(120,120,180,0.22);
    border-color: rgba(130,100,255,0.6);
    transform: translateX(3px);
}

/* New Chat button — shimmer sweep on hover */
[data-testid="stSidebar"] .stButton:first-of-type > button {
    background: linear-gradient(
        90deg,
        rgba(130,100,255,0.15) 0%,
        rgba(255,255,255,0.25) 50%,
        rgba(130,100,255,0.15) 100%
    );
    background-size: 200% auto;
    border: 1px solid rgba(130,100,255,0.45);
    font-weight: 600;
    color: inherit;
}
[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
    animation: shimmer 1.2s linear infinite;
    transform: none;
}

/* Active-thread button glow pulse */
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:focus,
[data-testid="stSidebar"] .stButton > button:active {
    animation: glowPulse 1.8s ease-in-out infinite;
    border-left: 3px solid rgba(130,100,255,0.9);
}

/* ═══════════════════════════════════════════════════
   EMPTY STATE
═══════════════════════════════════════════════════ */
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
    pointer-events: none;
}
.empty-state .bot-icon {
    font-size: 4rem;
    display: inline-block;
    animation: floatBob 3s ease-in-out infinite;
}
.empty-state h3 {
    margin: 0.8rem 0 0.3rem;
    font-size: 1.4rem;
    opacity: 0.75;
}
.empty-state p {
    opacity: 0.45;
    font-size: 0.95rem;
}

/* ═══════════════════════════════════════════════════
   TYPING INDICATOR (three bouncing dots)
═══════════════════════════════════════════════════ */
.typing-dots {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 0.4rem 0.2rem;
}
.typing-dots span {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: rgba(130,100,255,0.8);
    animation: bounce 1.2s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

/* ═══════════════════════════════════════════════════
   FOOTER
═══════════════════════════════════════════════════ */
.footer {
    position: fixed;
    bottom: 3.8rem;
    right: 1.2rem;
    font-size: 0.7rem;
    opacity: 0.3;
    pointer-events: none;
    letter-spacing: 0.04em;
}

/* ═══════════════════════════════════════════════════
   TOOL STATUS CARD
═══════════════════════════════════════════════════ */
.tool-status-card {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin: 0.1rem 0 0.75rem;
    padding: 0.65rem 0.8rem;
    border-radius: 12px;
    border: 1px solid rgba(130,100,255,0.35);
    background: linear-gradient(
        135deg,
        rgba(130,100,255,0.16) 0%,
        rgba(96,165,250,0.10) 100%
    );
    animation: statusAppear 0.25s ease;
}
.tool-status-icon {
    width: 1.35rem;
    height: 1.35rem;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
    font-weight: 700;
    flex-shrink: 0;
}
.tool-status-title {
    font-size: 0.83rem;
    font-weight: 600;
    line-height: 1.1rem;
}
.tool-status-meta {
    font-size: 0.72rem;
    opacity: 0.75;
    line-height: 0.95rem;
}
.tool-status-running .tool-status-icon {
    color: #93c5fd;
    border: 1px solid rgba(96,165,250,0.55);
    background: rgba(96,165,250,0.16);
    animation: statusPulse 1.4s infinite;
}
.tool-status-complete {
    border-color: rgba(74,222,128,0.35);
    background: linear-gradient(
        135deg,
        rgba(74,222,128,0.14) 0%,
        rgba(130,100,255,0.08) 100%
    );
}
.tool-status-complete .tool-status-icon {
    color: #86efac;
    border: 1px solid rgba(74,222,128,0.55);
    background: rgba(74,222,128,0.15);
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def render_tool_status(tool_name: str, state: str) -> str:
    safe_tool_name = html.escape(str(tool_name or "tool"))

    if state == "running":
        status_class = "tool-status-running"
        icon = "●"
        title = f"Running {safe_tool_name}"
        meta = "Executing tool request"
    else:
        status_class = "tool-status-complete"
        icon = "✓"
        title = f"{safe_tool_name} complete"
        meta = "Result merged into this response"

    return (
        f'<div class="tool-status-card {status_class}">'
        f'<span class="tool-status-icon">{icon}</span>'
        '<div class="tool-status-text">'
        f'<div class="tool-status-title">{title}</div>'
        f'<div class="tool-status-meta">{meta}</div>'
        "</div>"
        "</div>"
    )


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
selected_thread = None

# ============================ Sidebar ============================
with st.sidebar:
    st.markdown("## 🤖 LangGraph PDF Chatbot")
    st.caption("Powered by LangGraph · RAG · Streamlit")
    st.divider()
    st.caption(f"Thread: `{thread_key}`")

    if st.button("＋  New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        st.success(
            f"Using `{latest_doc.get('filename')}` "
            f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
        )
    else:
        st.info("No PDF indexed yet.")

    uploaded_pdf = st.file_uploader("Upload a PDF for this chat", type=["pdf"])
    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            st.info(f"`{uploaded_pdf.name}` already processed for this chat.")
        else:
            with st.status("Indexing PDF...", expanded=True) as status_box:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

    st.markdown("#### 💬 My Conversations")

    all_threads = st.session_state["chat_threads"][::-1]
    total = len(all_threads)

    if not all_threads:
        st.caption("No past conversations yet.")
    else:
        for i, thread_id in enumerate(all_threads):
            chat_number = total - i
            is_active = thread_id == st.session_state["thread_id"]
            label = f"{'▶ ' if is_active else ''}Chat {chat_number}"

            if st.button(label, key=f"side-thread-{thread_id}", use_container_width=True):
                selected_thread = thread_id

# ============================ Main Layout ========================
st.markdown(
    '<p class="gradient-title">🤖 Multi Utility Chatbot</p>',
    unsafe_allow_html=True,
)

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.caption(f"Thread: `{thread_key}` · Document: `{latest_doc.get('filename')}`")
else:
    st.caption(f"Thread: `{thread_key}`")

st.divider()

if not st.session_state["message_history"]:
    st.markdown(
        "<div class=\"empty-state\">"
        "<span class=\"bot-icon\">🤖</span>"
        "<h3>Start a conversation</h3>"
        "<p>Upload a PDF in the sidebar and ask me questions about it.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools...")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {
            "placeholder": st.empty(),
            "used_tool": False,
            "last_tool_name": "tool",
        }
        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            '<div class="typing-dots"><span></span><span></span><span></span></div>',
            unsafe_allow_html=True,
        )

        def ai_only_stream():
            started_assistant_tokens = False

            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    status_holder["used_tool"] = True
                    status_holder["last_tool_name"] = tool_name
                    status_holder["placeholder"].markdown(
                        render_tool_status(tool_name, "running"),
                        unsafe_allow_html=True,
                    )

                if isinstance(message_chunk, AIMessage):
                    if not started_assistant_tokens:
                        typing_placeholder.empty()
                        started_assistant_tokens = True
                    yield message_chunk.content

            typing_placeholder.empty()

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["used_tool"]:
            status_holder["placeholder"].markdown(
                render_tool_status(status_holder["last_tool_name"], "complete"),
                unsafe_allow_html=True,
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.markdown('<div class="footer">LangGraph · RAG · Streamlit</div>', unsafe_allow_html=True)

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        temp_messages.append({"role": role, "content": content})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
