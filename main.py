import streamlit as st
import tempfile
import chromadb
import aisuite as ai
from dotenv import load_dotenv
from llama_index.packs.raptor import RaptorPack
import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv()


st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by Groq")

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
    if not uploaded_file:
        st.info("Please upload a PDF file to continue.")
        st.stop()

    # Initialize Groq client
    client = ai.Client()

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
            
            nest_asyncio.apply()

            documents = SimpleDirectoryReader(input_files=[temp_file_path]).load_data()

            settings=Settings
            settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            from chromadb.config import Settings
            chroma_client = chromadb.PersistentClient(path="/tmp/.chroma", settings=Settings())
            embedding_model_name = "BAAI/bge-small-en-v1.5"  # Choose your desired model here
            hf_embeddings = HuggingFaceEmbedding(model_name=embedding_model_name)
            vector_store = ChromaVectorStore(
                chroma_collection=chroma_client.get_or_create_collection(name='persistent_collection'),
                embedding_function=hf_embeddings)
            raptor_pack = RaptorPack(
                documents,
                embed_model=settings.embed_model,
                llm=Groq(model="llama3-70b-8192", temperature=0.1),
                vector_store=vector_store,  # used for storage
                similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
                mode="tree_traversal",  # sets default mode
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
        You will be given some information about a course and you need to use that to explain a user's prompt.
        If they ask about a course, give it's id as well for ex: Tell me about a course in AI: One such course is CS XXX
        (replace with an actual number here) and instructors"""
    }

    # Combine system message, context, and conversation history for LLM input
    messages_for_llm = [system_message] + st.session_state.messages + [{"role": "user", "content": context + "\n" + prompt}]

    # Generate a response from Groq's model using context and conversation history
    chat_completion = client.chat.completions.create(
        messages=messages_for_llm,
        model="groq:llama3-70b-8192"  # Specify your desired model here
    )

    # Extract and display assistant's response
    msg = chat_completion.choices[0].message.content  # Adjust based on actual response structure from Groq API
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    
    # Create a clickable button
    # Source documents
    with st.expander("Source (summarized by LLM)"):
        st.write(context)

    