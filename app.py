import os
import dotenv
import base64
import streamlit as st
import asyncio
import edge_tts
from streamlit_mic_recorder import speech_to_text
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
from vector_db_pipeline import VectorDBPipeline  # Import your vector DB pipeline for adding new files

# Load environment variables
dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

# Available voices for Text-to-Speech (optional, keeping from original app1.py)
voices = {
    "William": "en-AU-WilliamNeural",
    "James": "en-PH-JamesNeural",
    "Jenny": "en-US-JennyNeural",
    "US Guy": "en-US-GuyNeural",
}

st.set_page_config(page_title="Aurora Health Chatbot", layout="wide", page_icon="./assets/logo-.png")  # Update icon if needed

# Title
# Title
st.markdown("""
    <h1 style='text-align: center;'>
        <span style='color: #0B3C9D;'>Aurora</span> 
        <span style='color: #1CA9E6;'>Health</span>
        <span style='color: #1ED5A6;'>Chatbot</span>
    </h1>
""", unsafe_allow_html=True)





# Streamlit setup
with st.sidebar:
    st.image("assets/logo.jpg", use_container_width=True)  # Update logo if needed
    st.markdown("## Aurora Health Chatbot")
    st.write("This bot can answer questions related to your health records, lab results, and medical history based on uploaded documents.")
    st.divider()

    # File uploader to add more files to the Vector DB
    uploaded_file = st.file_uploader("Upload PDF or Excel to add to your health records:", type=["pdf", "xlsx", "xls"])
    if uploaded_file:
        with st.spinner("Processing and adding file to database..."):
            # Save uploaded file temporarily
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Initialize pipeline and add the new file (assuming data_folder is temporary for new uploads)
            pipeline = VectorDBPipeline(data_folder=temp_dir, index_name="health-docs")  # Use the correct index name from your vector_db_pipeline.py
            pipeline.run_pipeline()  # This will process and add the new file to Pinecone
            st.success(f"File '{uploaded_file.name}' added to the database successfully!")

# Load vectorstore only once (Pinecone with OpenAI embeddings)
if "vectorstore" not in st.session_state:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Changed to match index dimension 1536
    st.session_state["vectorstore"] = PineconeVectorStore(
        index_name="health-docs",  # Use the correct index name from your vector_db_pipeline.py
        embedding=embeddings
    )

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = [
        {"role": "assistant", "content": "Hello! I'm Aurora, your personal health assistant. How can I help with your medical records today?"}
    ]

def format_docs(docs):
    return "\n\n".join(
        [f'Document {i+1}:\n{doc.page_content}\n'
         f'Source: {doc.metadata.get("source", "Unknown")}\n'
         f'Category: {doc.metadata.get("category", "Unknown")}\n'
         f'Page/Sheet: {doc.metadata.get("page_number", doc.metadata.get("sheet_name", "N/A"))}\n-------------'
         for i, doc in enumerate(docs)]
    )

# def format_metadata(docs):
#     """Format metadata for display in a table"""
#     metadata_list = []
#     for i, doc in enumerate(docs):
#         metadata = {
#             "Document": i + 1,
#             "File Name": doc.metadata.get("file_name", "Unknown"),
#             "File Type": doc.metadata.get("file_type", "Unknown"),
#             "Page/Sheet": doc.metadata.get("page_number", doc.metadata.get("sheet_name", "N/A")),
#             "Source Path": doc.metadata.get("source", "Unknown"),
#             "Category": doc.metadata.get("category", "Unknown"),
#             "Created Date": doc.metadata.get("created_date", "N/A"),
#             "Modified Date": doc.metadata.get("modified_date", "N/A"),
#             "File Size (bytes)": doc.metadata.get("file_size", "N/A"),
#         }
#         if doc.metadata.get("file_type") == "pdf":
#             metadata["Total Pages"] = doc.metadata.get("total_pages", "N/A")
#         elif doc.metadata.get("file_type") == "excel":
#             metadata["Sheet Index"] = doc.metadata.get("sheet_index", "N/A")
#             metadata["Total Sheets"] = doc.metadata.get("total_sheets", "N/A")
#             metadata["Rows Count"] = doc.metadata.get("rows_count", "N/A")
#             metadata["Columns Count"] = doc.metadata.get("columns_count", "N/A")
#         metadata_list.append(metadata)
#     return metadata_list

# Reset conversation
def reset_conversation():
    st.session_state.pop('chat_history')
    st.session_state['chat_history'] = [
        {"role": "assistant", "content": "Hello! I'm Aurora, your personal health assistant. How can I help with your medical records today?"}
    ]

def rag_qa_chain(question, retriever, chat_history):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)  ## You can change the model name
 
    output_parser = StrOutputParser()

    # System prompt to contextualize the question
    contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    
    contextualize_q_chain = contextualize_q_prompt | llm | output_parser

    # Professional prompt based on sample interactions: Detailed, tabular comparisons, emojis for status, health-focused
    qa_system_prompt = """You are Aurora, a professional AI health assistant specialized in analyzing and summarizing medical records, lab results, and health data.
    Use the retrieved documents as context to answer the user's question accurately and professionally. Structure your responses like a medical consultant:
    - Use tables for comparisons, trends, or breakdowns (e.g., date, test type, result, comment).
    - Include emojis for quick status indicators: ✅ for normal/good, ❌ for high/risk, ⚠ for borderline.
    - Provide interpretations, trends, and summaries where relevant.
    - If comparing results, include a summary of trends and suggestions for improvement.
    - Keep responses concise, clear, and empathetic. Always suggest consulting a doctor for advice.
    - If the information is not in the context, politely say: "I'm sorry, I don't have that information in your records. Please upload relevant documents or consult your doctor."
    - If the question is irrelevant to health or medical records, steer back: "I'm here to help with your health records. Could you ask about your lab results or medical history?"
    - Respond in English only.
    - Ensure all responses come from the vector database, do not provide answers from external knowledge sources.

    Retrieved Documents (Context):
    ------------
    {context}
    ------------
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def retrieve_docs_and_response(input_dict):
        context = contextualize_q_chain.invoke({"question": input_dict["question"], "chat_history": input_dict["chat_history"]})
        docs = retriever.invoke(context)
        formatted_context = format_docs(docs)
        response = (prompt | llm | output_parser).invoke({
            "question": input_dict["question"],
            "chat_history": input_dict["chat_history"],
            "context": formatted_context
        })
        return {"response": response, "docs": docs}

    rag_chain = (
        RunnablePassthrough.assign(
            result=retrieve_docs_and_response
        )
    )
    
    return rag_chain.stream({"question": question, "chat_history": chat_history})

# Generate the speech from text (optional TTS)
async def generate_speech(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        await communicate.save(temp_file.name)
        temp_file_path = temp_file.name
    return temp_file_path

# Get audio player
def get_audio_player(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        return f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">'
        
# Text-to-Speech function
def generate_voice(text, voice):
    text_to_speak = (text).translate(str.maketrans('', '', '#-*_😊👋😄😁🥳👍🤩😂😎'))  # Remove special chars
    with st.spinner("Generating voice response..."):
        temp_file_path = asyncio.run(generate_speech(text_to_speak, voice)) 
        audio_player_html = get_audio_player(temp_file_path)
        st.markdown(audio_player_html, unsafe_allow_html=True)
        os.unlink(temp_file_path)

# Sidebar voice option (optional)
if st.sidebar.toggle("Enable Voice Response"):
    voice_option = st.sidebar.selectbox("Choose a voice for response:", options=list(voices.keys()), key="voice_response")

# Main interface columns
col1, col2 = st.columns([1, 5])

# Displaying chat history
for message in st.session_state.chat_history:
    avatar = "assets/user.png" if message["role"] == "user" else "assets/assistant.png"
    with col2:
        st.chat_message(message["role"], avatar=avatar).write(message["content"])

# Handle voice or text input
with col1:
    st.button("Reset", use_container_width=True, on_click=reset_conversation)

    with st.spinner("Converting speech to text..."):
        text = speech_to_text(language="en", just_once=True, key="STT", use_container_width=True)

query = st.chat_input("Type your question about your health records:")

# Generate the response
if text or query:
    col2.chat_message("user", avatar="assets/user.png").write(text if text else query)
    
    st.session_state.chat_history.append({"role": "user", "content": text if text else query})

    # Generate response
    with col2.chat_message("assistant", avatar="assets/assistant.png"):
        try:
            stream = rag_qa_chain(
                question=text if text else query,
                retriever=st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 6}),
                chat_history=st.session_state.chat_history
            )
            response = ""
            docs = []
            for chunk in stream:
                response = chunk.get("result", {}).get("response", "")
                docs = chunk.get("result", {}).get("docs", [])
            
            # Display the response
            st.write(response)
            
            # Display metadata in an expander
            # if docs:
            #     with st.expander("View Source Documents and Metadata"):
            #         metadata_list = format_metadata(docs)
            #         st.table(metadata_list)
            
            # Add response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An internal error occurred: {str(e)}. Please check your API keys and internet connection.")

    # Generate voice response if enabled
    if "voice_response" in st.session_state and st.session_state.voice_response:
        response_voice = st.session_state.voice_response

        generate_voice(response, voices[response_voice])

