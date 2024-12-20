from openai import OpenAI
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Streamlit Sidebar for API Key Input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/Riddhimaan-Senapati/CourseAdvisor)"

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

# File uploader for document input
uploaded_file = st.file_uploader("Upload a PDF article", type="pdf")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Process user input and generate responses
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Handle document embeddings if a file is uploaded
    if uploaded_file is not None:
        # Read the uploaded PDF file content
        loader = PyPDFLoader(uploaded_file.name)
        docs = loader.load()

        # Create embeddings for the document using OpenAI's embedding model
        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False)
        splits = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key,model="text-embedding-3-large")
        # Store the embedding temporarily in ChromaDB
        vector_store = Chroma(embedding_function=embeddings)
        _ = vector_store.add_documents(documents=splits)


    # Create a retriever from the vector store
    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(prompt)

    # Prepare context for the LLM response
    context = "\n\n".join(doc for doc in retrieved_docs)

    # Define a system message to guide the LLM's behavior
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant knowledgeable about the content of the uploaded PDF document."
    }

    # Combine system message, context, and conversation history for LLM input
    messages_for_llm = [system_message] + st.session_state.messages + [{"role": "user", "content": context + "\n" + prompt}]

    # Generate a response from OpenAI's model using context and conversation history
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages_for_llm)
    
    # Extract and display assistant's response
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

