import os
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient
load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
def build_faiss_vectorstore():
    print("Loading document...")
    loader = TextLoader("data.txt")
    docs = loader.load()
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    print("Creating FAISS vector DB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
vector_store = build_faiss_vectorstore()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
@tool
def retrieve_from_document(query: str) -> str:
    """Strictly retrieve information from the uploaded notes only."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    if not retrieved_docs:
        return "NO_DOCUMENT_MATCH"
    serialized = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return serialized
# --------------------------------------------------
# 5. Tavily Search Tool
# --------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Use Tavily only when document has no relevant information."""
    response = tavily.search(query=query, max_results=3)
    if not response or not response.get("results"):
        return "NO_WEB_RESULT"
    summary = "\n\n".join(
        f"- {item['title']}: {item['content']}" for item in response["results"]
    )
    return summary
tools = [retrieve_from_document, web_search]
# --------------------------------------------------
# 6. Prompt with strict logic: Document → Web → Nothing
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an Agentic RAG assistant.
 
STRICT RULES:
1. ALWAYS check the document first using the `retrieve_from_document` tool.
2. If document returns "NO_DOCUMENT_MATCH", then and only then use `web_search`.
3. If both fail, respond: "The information is not available in the document."
4. Do NOT hallucinate.
5. Do NOT answer from general knowledge unless Tavily specifically provides it.
6. Prefer document answers over Tavily answers.
Follow this exact order:
DOCUMENT → WEB (fallback) → "not in document"
""",
        ),
        # MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
# --------------------------------------------------
# 7. Create Agent & Executor
# --------------------------------------------------
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True,
)
# --------------------------------------------------
# 8. Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("🦜 Agentic RAG with Tavily Fallback (Strict RAG Mode)")
if "messages" not in st.session_state:
    st.session_state.messages = []
# Show chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)
# User input
user_question = st.chat_input("Ask anything...")
if user_question:
    # Add user message
    user_msg = HumanMessage(user_question)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(user_question)
    # Invoke agent
    result = agent_executor.invoke(
        {
            "input": user_question,
            "chat_history": st.session_state.messages,
        }
    )
    print("Agent final output:", result)
    ai_text = result["output"]
    # Show assistant reply
    with st.chat_message("assistant"):
        st.markdown(ai_text[0]['text']+'\n'+ai_text[1])
    st.session_state.messages.append(AIMessage(ai_text[0]['text']+'\n'+ai_text[1]))
 
