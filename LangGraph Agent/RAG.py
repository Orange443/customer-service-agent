from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fix tokenizer warning

# Initialize the LLM and embeddings
llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
pdf_path = "Stock_Market_Performance_2024.pdf"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Load and process PDF
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"The specified PDF file does not exist: {pdf_path}")

try:
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load()
    print(f"PDF loaded successfully with {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunk documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pages_split = text_splitter.split_documents(pages)

# Set up Chroma vector store
persist_directory = r"/Users/adityakapadia/My PC/LangGraph Agent"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # Check if Chroma database exists; load it if present, else create new
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        print("Loaded existing Chroma vector store")
    except Exception:
        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print("Created new Chroma vector store")
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

@tool
def retriever_tool(query: str) -> str:
    """
    Searches and returns information from the Stock Market Performance 2024 document.
    """
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the Stock Market Performance 2024 document."
        results = [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
        return "\n\n".join(results)
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

# Bind tools to LLM
tools = [retriever_tool]
llm = llm.bind_tools(tools)
tools_dict = {tool.name: tool for tool in tools}

# Define system prompt
system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool to fetch relevant information. You can make multiple calls if needed.
Cite specific parts of the documents in your answers.
"""

# Define nodes
def call_llm(state: AgentState) -> AgentState:
    """Call the LLM with the current state."""
    messages = [SystemMessage(content=system_prompt)] + list(state.messages)
    message = llm.invoke(messages)
    return {"messages": [message]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        if t["name"] not in tools_dict:
            print(f"Tool: {t['name']} does not exist.")
            result = "Incorrect Tool Name. Please retry with available tools."
        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            print(f"Result length: {len(str(result))}")
        results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}

# Build and compile graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()

# Run the agent
def running_agent():
    print("\n=== RAG AGENT ===")
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)

if __name__ == "__main__":
    running_agent()