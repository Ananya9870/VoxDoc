import streamlit as st
import os, tempfile, uuid, asyncio
from dotenv import load_dotenv
from gtts import gTTS
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from fastembed import TextEmbedding

# Load custom Agent/Runner from your agents.py
from agents import Agent, Runner

load_dotenv()
COLLECTION_NAME = "voice-rag-agent"

def init_session_state():
    """Initialize multi-chat session state."""
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = {}
    
    if "current_chat_id" not in st.session_state:
        new_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_id
        st.session_state.all_chats[new_id] = {"name": "New Chat", "messages": []}
        
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = []
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False

def setup_qdrant():
    """Local Qdrant setup."""
    client = QdrantClient(path="./qdrant_db")
    embedding_model = TextEmbedding()
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    except Exception:
        pass
    return client, embedding_model

def main():
    st.set_page_config(page_title="Voice RAG", page_icon="🎙️", layout="wide")
    init_session_state()

    # --- SIDEBAR: CHAT HISTORY ---
    with st.sidebar:
        st.title("📂 Chat History")
        
        if st.button("➕ New Chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.all_chats[new_id] = {"name": "New Chat", "messages": []}
            st.session_state.current_chat_id = new_id
            st.rerun()

        st.divider()
        
        # Display list of past chats
        for chat_id, data in reversed(list(st.session_state.all_chats.items())):
            is_active = (chat_id == st.session_state.current_chat_id)
            if st.button(data["name"], key=chat_id, use_container_width=True, 
                         type="primary" if is_active else "secondary"):
                st.session_state.current_chat_id = chat_id
                st.rerun()

    # --- MAIN CHAT AREA ---
    st.title("🎙️ AI Doc Agent")
    
    current_chat = st.session_state.all_chats[st.session_state.current_chat_id]

    if not st.session_state.setup_complete:
        if st.button("🚀 Initialize System"):
            st.session_state.setup_complete = True
            st.rerun()

    if st.session_state.setup_complete:
        # Step 2: Upload PDF
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file and uploaded_file.name not in st.session_state.processed_documents:
            with st.spinner("Analyzing Document..."):
                client, model = setup_qdrant()
                with tempfile.NamedTemporaryFile(delete=False) as tf:
                    tf.write(uploaded_file.getbuffer())
                    loader = PyPDFLoader(tf.name)
                    docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)

                for chunk in chunks:
                    vector = list(model.embed([chunk.page_content]))[0].tolist()
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"content": chunk.page_content})]
                    )
                st.session_state.processed_documents.append(uploaded_file.name)
                st.success("PDF Indexed!")

        # Display history for the current selected chat session
        for message in current_chat["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Step 3: Chat Input
        query = st.chat_input("Ask about your PDF...")
        if query:
            # Name the chat based on first query
            if current_chat["name"] == "New Chat":
                current_chat["name"] = query[:25] + "..."

            with st.chat_message("user"):
                st.write(query)
            current_chat["messages"].append({"role": "user", "content": query})

            with st.spinner("Searching & Thinking..."):
                client, model = setup_qdrant()
                query_vector = list(model.embed([query]))[0].tolist()

                # Retrieval from Qdrant
                search_results = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    limit=3
                ).points

                context = "\n".join([r.payload['content'] for r in search_results if r.payload])
                
                # Context from last 5 messages of THIS chat
                memory_context = "\n".join([f"{m['role']}: {m['content']}" for m in current_chat["messages"][-5:]])

                rag_agent = Agent(name="Guide", instructions="Use the context and history to answer concisely.")

                # Async Execution
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                prompt = f"History:\n{memory_context}\n\nContext:\n{context}\n\nUser: {query}"
                result = loop.run_until_complete(Runner.run(rag_agent, prompt))
                response_text = result.final_output

                current_chat["messages"].append({"role": "assistant", "content": response_text})

                with st.chat_message("assistant"):
                    st.write(response_text)

                    # Voice Output
                    tts = gTTS(text=response_text, lang='en')
                    audio_path = f"speech_{uuid.uuid4()}.mp3"
                    tts.save(audio_path)
                    st.audio(audio_path, autoplay=True)

if __name__ == "__main__":
    main()