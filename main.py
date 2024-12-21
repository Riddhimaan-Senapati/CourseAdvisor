import streamlit as st
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import chromadb
from chromadb.config import Settings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import aisuite as ai
from dotenv import load_dotenv
load_dotenv()


st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by Groq")

# File uploader for document input
uploaded_file = st.file_uploader("Upload a PDF article", type="pdf")

# Initialize message history and vector store in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

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
    if st.session_state.vector_store is None:
        with st.spinner("Processing PDF File... This may take a while ⏳"):
            # Save the uploaded PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            
            # Load the PDF using PyPDFLoader from the temporary file path
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

            # Split the documents into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False)
            
            splits = text_splitter.split_documents(docs)

            # Initialize Hugging Face Embeddings model
            model_name = "sentence-transformers/all-mpnet-base-v2"  # Choose your desired model here
            hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)

            # Generate embeddings for the document splits
            document_texts = [split.page_content for split in splits]
            embeddings = hf_embeddings.embed_documents(document_texts)

            # Store the embedding temporarily in ChromaDB
            chroma_client = chromadb.PersistentClient(
                path="/tmp/.chroma",
                settings=Settings()
            )

            vector_store = Chroma(
                embedding_function=hf_embeddings,  # Pass embed_documents method directly
                client=chroma_client  # Pass the persistent client here
            )
            
            _ = vector_store.add_documents(documents=splits)  # Add documents directly

            # Save vector store to session state for future use
            st.session_state.vector_store = vector_store

        st.success("PDF processed successfully!")

    # Create a retriever from the vector store and get relevant documents based on user input
    retriever = st.session_state.vector_store.as_retriever()
    retrieved_docs = retriever.invoke(prompt)

    # Prepare context for the LLM response
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Define a system message to guide the LLM's behavior
    system_message = {
        "role": "system",
        "content": """You are a course advisor meant to help students choose the courses that they are most suitable for.
        You will be given some information about a course and you need to use that to explain a user's prompt.
        If they ask about a course, give it's id as well for ex: Tell me about a course in AI: One such course is CS XXX
        (replace with an actual number here)"""
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
    with st.expander("Source documents"):
        st.write(context)

    