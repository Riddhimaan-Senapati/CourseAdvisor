import streamlit as st
import tempfile
import chromadb
from dotenv import load_dotenv
from llama_index.packs.raptor import RaptorPack
import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
import os

load_dotenv()

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="feedback_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


st.title("Course Advisor")
st.caption("🚀 A Streamlit chatbot powered by OpenAI and RAPTOR RAG")

# File uploader for document input
uploaded_file = st.file_uploader("Upload a PDF article", type="pdf")

# Initialize message history and vector store in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "raptor_pack" not in st.session_state:
    st.session_state["raptor_pack"] = None

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Process user input and generate responses
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    if not uploaded_file:
        st.info("Please upload a PDF file to continue.")
        st.stop()

    # Initialize OpenAI client using aisuite library
    os.environ["OPENAI_API_KEY"] = openai_api_key
    client = OpenAI()

    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Check if vector store is already created; if not, process the uploaded PDF
    if st.session_state.raptor_pack is None:
        with st.spinner("Processing PDF File... This may take a while ⏳"):
            # Save the uploaded PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name

            # Constructing the Clusters/Hierarchy Tree
            
            nest_asyncio.apply()

            documents = SimpleDirectoryReader(input_files=[temp_file_path]).load_data()

            from chromadb.config import Settings
            chroma_client = chromadb.PersistentClient(path="/tmp/.chroma", settings=Settings())

            vector_store = ChromaVectorStore(
                chroma_collection=chroma_client.get_or_create_collection(name='persistent_collection'),
                embedding_function=OpenAIEmbedding(model="text-embedding-3-small"))
            
            raptor_pack = RaptorPack(
                documents,
                embed_model=OpenAIEmbedding(model="text-embedding-3-small"), 
                llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
                vector_store=vector_store,  # used for storage
                similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
                mode="collapsed",  # sets default mode
                transformations=[SentenceSplitter(chunk_size=2000, chunk_overlap=100)]) # transformations applied for ingestion
            
            st.session_state.raptor_pack=raptor_pack


        st.success("PDF processed successfully!")

    # Prepare context for the LLM response

    nodes = st.session_state.raptor_pack.run(prompt, mode="tree_traversal")
    context="\n".join([node.text for node in nodes])

    # Define a system message to guide the LLM's behavior
    system_message = {
        "role": "system",
        "content": """You are a course advisor meant to help students choose the courses that they are most suitable for.
        You will be given some information about a course as context and you need to use that to explain a user's prompt.
        If they ask you that is not course related. Respond with "Sorry,I can't help you". Do not respond to anything that
        is not course related. Always in that case. Respond with "Sorry,I can't help you". """
    }

    # Combine system message, context, and conversation history for LLM input
    messages_for_llm = [system_message] + st.session_state.messages + [{"role": "user", "content": context + "\n" + prompt}]

    # Generate a response from Groq's model using context and conversation history
    chat_completion = client.chat.completions.create(
        messages=messages_for_llm,
        model="openai:gpt-3.5-turbo"  # Specify your desired model here
    )

    # Extract and display assistant's response
    msg = chat_completion.choices[0].message.content  # Adjust based on actual response structure from Open AI API
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    
    # Create a clickable button to show Source documents
    with st.expander("Source (summarized by LLM)"):
        st.write(context)

    