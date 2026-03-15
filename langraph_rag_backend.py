from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2",model_kwargs={"device": "cpu"})

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _normalize_thread_id(thread_id: Optional[str]) -> Optional[str]:
    if thread_id is None:
        return None
    value = str(thread_id).strip()
    return value or None


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    normalized_thread_id = _normalize_thread_id(thread_id)
    if normalized_thread_id and normalized_thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[normalized_thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    normalized_thread_id = _normalize_thread_id(thread_id)
    if not normalized_thread_id:
        raise ValueError("Invalid thread_id for PDF ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[normalized_thread_id] = retriever
        _THREAD_METADATA[normalized_thread_id] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Thread context is injected automatically from runtime config.
    A thread_id may also be provided by the model if available.
    """
    runtime_thread_id = None
    if config and isinstance(config, dict):
        runtime_thread_id = config.get("configurable", {}).get("thread_id")

    # Always trust runtime thread context first; model-supplied IDs can be wrong.
    effective_thread_id = runtime_thread_id or thread_id
    effective_thread_id = _normalize_thread_id(effective_thread_id)

    retriever = _get_retriever(effective_thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(effective_thread_id or "", {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    thread_id = _normalize_thread_id(thread_id)

    doc_meta = _THREAD_METADATA.get(str(thread_id), {})
    if doc_meta:
        doc_context = f"An uploaded PDF document named '{doc_meta.get('filename')}' is current ACTIVE and AVAILABLE. You MUST use the `rag_tool` to interact with or read this document before answering."
    else:
        doc_context = "No document is currently uploaded."

    system_message = SystemMessage(
        content=(
            f"You are a helpful assistant. {doc_context}\n"
            "Thread context is provided automatically to tools. You can also use "
            "the web search, stock price, and calculator tools when helpful."
        )
    )

    messages = [system_message, *state["messages"]]

    def _last_user_text() -> str:
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                if isinstance(message.content, str):
                    return message.content
                return str(message.content)
        return ""

    def _invoke_without_tools_fallback() -> AIMessage:
        user_text = _last_user_text()
        retriever = _get_retriever(thread_id)
        context_sections = []

        if user_text and retriever is not None:
            try:
                docs = retriever.invoke(user_text)
                for idx, doc in enumerate(docs[:4], 1):
                    snippet = doc.page_content.strip().replace("\n", " ")
                    if len(snippet) > 900:
                        snippet = snippet[:900] + "..."
                    context_sections.append(f"[{idx}] {snippet}")
            except Exception:
                context_sections = []

        fallback_system = SystemMessage(
            content=(
                "Tool calling is temporarily unavailable for this turn. "
                "Answer directly without calling tools. If provided context is "
                "insufficient, say so briefly."
            )
        )

        fallback_messages = [fallback_system]
        if context_sections:
            source_file = _THREAD_METADATA.get(str(thread_id), {}).get("filename", "uploaded PDF")
            fallback_messages.append(
                SystemMessage(
                    content=(
                        f"Retrieved context from {source_file}:\n"
                        + "\n\n".join(context_sections)
                    )
                )
            )

        fallback_messages.extend(state["messages"])
        result = llm.invoke(fallback_messages, config=config)
        if isinstance(result, AIMessage):
            return result
        return AIMessage(content=str(result))

    try:
        response = llm_with_tools.invoke(messages, config=config)

        # Guardrail: if a retriever exists but the model still asks for upload,
        # answer using direct retrieval fallback so users don't see false negatives.
        if _get_retriever(thread_id) is not None and isinstance(response, AIMessage):
            response_text = (
                response.content.lower()
                if isinstance(response.content, str)
                else str(response.content).lower()
            )
            if (
                "no document" in response_text
                or "no pdf" in response_text
                or ("upload" in response_text and "pdf" in response_text)
            ):
                response = _invoke_without_tools_fallback()
    except Exception as exc:
        error_text = str(exc)
        is_tool_call_error = (
            "Failed to call a function" in error_text
            or "tool_use_failed" in error_text
        )

        if not is_tool_call_error:
            raise

        retry_instruction = SystemMessage(
            content=(
                "If you call a tool, arguments must strictly match the tool schema as "
                "valid JSON with only defined keys."
            )
        )

        try:
            response = llm_with_tools.invoke(
                [retry_instruction, *messages],
                config=config,
            )
        except Exception:
            try:
                response = _invoke_without_tools_fallback()
            except Exception:
                response = AIMessage(
                    content=(
                        "I hit a temporary tool-calling issue. Please ask again in a "
                        "slightly simpler way."
                    )
                )

    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    normalized_thread_id = _normalize_thread_id(thread_id)
    return bool(normalized_thread_id and normalized_thread_id in _THREAD_RETRIEVERS)


def thread_document_metadata(thread_id: str) -> dict:
    normalized_thread_id = _normalize_thread_id(thread_id)
    if not normalized_thread_id:
        return {}
    return _THREAD_METADATA.get(normalized_thread_id, {})
