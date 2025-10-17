import streamlit as st
from pathlib import Path
import tempfile
import os
import json
import uuid
import shutil
from datetime import datetime
from chunking import process_pdf
from embed_and_index import embed_chunks
from retriever import Retriever
from generator import Generator

# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(page_title="Multilingual RAG Chatbot", layout="wide")
st.title("📄 Multilingual RAG Chatbot")
st.markdown("Upload multiple PDFs and chat with the AI about their content.")

# ----------------------------
# Cleanup Management
# ----------------------------
def cleanup_previous_session_data():
    """Clean up data from previous sessions"""
    try:
        # Define directories to clean up
        data_dirs = [
            Path("data/chunks"),
            Path("data/faiss_index"),
            Path("data/metadata")
        ]
        
        for data_dir in data_dirs:
            if data_dir.exists():
                shutil.rmtree(data_dir)
                
        # Recreate directories
        for data_dir in data_dirs:
            data_dir.mkdir(parents=True, exist_ok=True)
            
    except Exception as e:
        st.sidebar.warning(f"Cleanup warning: {e}")

def cleanup_current_session_data():
    """Clean up data for current session when app closes or session ends"""
    if "session_cleanup_done" not in st.session_state:
        try:
            # Get list of PDFs processed in this session
            current_pdfs = list(st.session_state.get("processed_pdfs", {}).keys())
            
            for pdf_name in current_pdfs:
                # Clean up chunk directories
                chunk_dir = Path("data/chunks") / Path(pdf_name).stem
                if chunk_dir.exists():
                    shutil.rmtree(chunk_dir)
                
                # Clean up FAISS index files
                faiss_pattern = f"*{Path(pdf_name).stem}*"
                faiss_files = list(Path("data/faiss_index").glob(faiss_pattern))
                for faiss_file in faiss_files:
                    faiss_file.unlink()
                
                # Clean up metadata files
                metadata_pattern = f"*{Path(pdf_name).stem}*"
                metadata_files = list(Path("data/metadata").glob(metadata_pattern))
                for metadata_file in metadata_files:
                    metadata_file.unlink()
                    
            st.session_state.session_cleanup_done = True
            
        except Exception as e:
            pass  # Silent cleanup

# ----------------------------
# Chat History Management
# ----------------------------
def load_chat_history():
    """Load chat history from file"""
    try:
        if os.path.exists("chat_history.json"):
            with open("chat_history.json", "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        pass  # Silent loading
    return []

def save_chat_history(chat_history):
    """Save chat history to file"""
    try:
        with open("chat_history.json", "w", encoding="utf-8") as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        pass  # Silent save

# ----------------------------
# Initialize session state
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = load_chat_history()

if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]  # Short session ID

if "processed_pdfs" not in st.session_state:
    st.session_state["processed_pdfs"] = {}  # Store multiple PDF retrievers/generators

if "active_pdf" not in st.session_state:
    st.session_state["active_pdf"] = None

if "session_cleanup_done" not in st.session_state:
    st.session_state["session_cleanup_done"] = False

if "temp_files" not in st.session_state:
    st.session_state["temp_files"] = []  # Track temporary files for cleanup

# ----------------------------
# Cleanup previous session data on startup
# ----------------------------
if "initial_cleanup_done" not in st.session_state:
    cleanup_previous_session_data()
    st.session_state.initial_cleanup_done = True

# ----------------------------
# Sidebar: Upload PDF
# ----------------------------
with st.sidebar:
    st.header("📄 Document Processing")
    
    # Multiple PDF upload
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    process_button = st.button("Process PDF(s)")
    
    # Display processed PDFs
    if st.session_state.processed_pdfs:
        st.markdown("---")
        st.header("📚 Processed Documents")
        
        # PDF selector
        pdf_options = list(st.session_state.processed_pdfs.keys())
        selected_pdf = st.selectbox(
            "Select active document:",
            options=pdf_options,
            index=pdf_options.index(st.session_state.active_pdf) if st.session_state.active_pdf in pdf_options else 0
        )
        
        if selected_pdf and selected_pdf != st.session_state.active_pdf:
            st.session_state.active_pdf = selected_pdf
            st.rerun()
        
        # Show processed PDFs list
        for pdf_name in st.session_state.processed_pdfs.keys():
            status = "✅" if pdf_name == st.session_state.active_pdf else "📄"
            st.write(f"{status} {pdf_name}")
    
    # st.markdown("---")
    # st.header("💬 Session Info")
    
    # # Display session info
    # st.info(f"Session ID: {st.session_state.session_id}")
    # st.info(f"Active PDF: {st.session_state.active_pdf or 'None'}")
    # st.info(f"Total messages: {len(st.session_state.chat_history)}")

# ----------------------------
# Process Multiple PDFs
# ----------------------------
if uploaded_files and process_button:
    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.processed_pdfs:
            st.warning(f"📄 {uploaded_file.name} already processed. Skipping...")
            continue
            
        st.info(f"🔹 Processing {uploaded_file.name}...")
        
        # Create temporary file but don't delete it yet
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
        
        # Store temp file path for later cleanup
        st.session_state.temp_files.append(pdf_path)

        # Chunking
        try:
            chunk_output_dir = Path("data/chunks") / Path(uploaded_file.name).stem
            chunk_files, chunk_dir = process_pdf(pdf_path, output_base_dir=chunk_output_dir)
            st.success(f"✅ {uploaded_file.name} processed into {len(chunk_files)} chunks")
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            # Clean up temp file on error
            try:
                os.unlink(pdf_path)
                st.session_state.temp_files.remove(pdf_path)
            except:
                pass
            continue

        # Embedding + FAISS
        try:
            total_chunks, faiss_index_path, metadata_path = embed_chunks(chunk_dir)
            st.success(f"✅ Embedded {total_chunks} chunks from {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Error during embedding {uploaded_file.name}: {e}")
            # Clean up temp file on error
            try:
                os.unlink(pdf_path)
                st.session_state.temp_files.remove(pdf_path)
            except:
                pass
            continue

        # Initialize Retriever
        try:
            retriever = Retriever(
                chunk_folder=chunk_dir,
                faiss_index_path=faiss_index_path,
                metadata_path=metadata_path,
                semantic_weight=0.6,
                keyword_weight=0.4,
                top_k=10
            )
        except Exception as e:
            st.error(f"❌ Error initializing Retriever for {uploaded_file.name}: {e}")
            # Clean up temp file on error
            try:
                os.unlink(pdf_path)
                st.session_state.temp_files.remove(pdf_path)
            except:
                pass
            continue

        # Initialize Generator
        try:
            generator = Generator(use_reranker=True)
            
            # Store in processed PDFs
            st.session_state.processed_pdfs[uploaded_file.name] = {
                "retriever": retriever,
                "generator": generator,
                "processed_at": datetime.now().isoformat(),
                "chunk_dir": str(chunk_dir),
                "faiss_index_path": str(faiss_index_path),
                "metadata_path": str(metadata_path)
            }
            
            # Set as active if first PDF
            if st.session_state.active_pdf is None:
                st.session_state.active_pdf = uploaded_file.name
                
            st.success(f"✅ {uploaded_file.name} ready for chatting!")
            
            # Now we can safely delete the temp file after successful processing
            try:
                os.unlink(pdf_path)
                st.session_state.temp_files.remove(pdf_path)
            except:
                pass
            
        except Exception as e:
            st.error(f"❌ Error initializing Generator for {uploaded_file.name}: {e}")
            # Clean up temp file on error
            try:
                os.unlink(pdf_path)
                st.session_state.temp_files.remove(pdf_path)
            except:
                pass
            continue

# ----------------------------
# Clean up any remaining temp files
# ----------------------------
def cleanup_temp_files():
    """Clean up any remaining temporary files"""
    for temp_file in st.session_state.temp_files[:]:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            st.session_state.temp_files.remove(temp_file)
        except Exception as e:
            pass  # Silent cleanup

# ----------------------------
# Main Content Area with Two Columns
# ----------------------------
if st.session_state.processed_pdfs and st.session_state.active_pdf:
    # Create two columns: 70% for current chat, 30% for previous sessions
    col1, col2 = st.columns([7, 3])
    
    with col1:
        st.markdown("---")
        st.subheader(f"💬 Current Chat: {st.session_state.active_pdf}")
        
        # Get active PDF components - FIXED: Ensure we're getting the correct active PDF
        active_pdf_data = st.session_state.processed_pdfs.get(st.session_state.active_pdf)
        
        if active_pdf_data is None:
            st.error("No data found for active PDF. Please process a PDF first.")
            st.stop()
            
        retriever = active_pdf_data["retriever"]
        generator = active_pdf_data["generator"]

        # Display current session chats for active PDF
        current_session_chats = [
            chat for chat in st.session_state.chat_history 
            if chat.get("session_id") == st.session_state.session_id 
            and chat.get("pdf_name") == st.session_state.active_pdf
        ]
        
        if current_session_chats:
            for i, chat in enumerate(current_session_chats):
                st.markdown(f"**You:** {chat['user']}")
                st.markdown(f"**AI:** {chat['bot']}")
                
                # Create unique expander using the timestamp or index
                # expander_label = f"🔍 Show context for this answer"
                # with st.expander(expander_label):
                #     for j, c in enumerate(chat.get("context", [])):
                #         st.markdown(f"- **[{c['language']}]** {c['text'][:300]}... (Source: {c['source_file']})")
                #st.markdown("---")
        else:
            st.info("💡 Start a conversation by asking a question below!")

        # ----------------------------
        # Chat Input Form
        # ----------------------------
        st.markdown("---")
        
        # Use a form with a unique key and clear the input after submission
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask a question:",
                value="",  # Always start with empty value
                placeholder="Type your question here...",
                key="chat_input"  # Unique key for the input widget
            )
            submit_button = st.form_submit_button("Send")

        # Process the query outside the form to avoid double submission issues
        if submit_button and user_input.strip():
            query = user_input.strip()

            # Retrieve top chunks - FIXED: Using the correct retriever for active PDF
            try:
                retriever_results = retriever.hybrid_search(query)
                
                # Generate answer - FIXED: Using the correct generator for active PDF
                answer = generator.generate_answer(query, retriever_results)

                # Create chat entry with metadata
                chat_entry = {
                    "user": query,
                    "bot": answer,
                    "context": retriever_results[:5],
                    "timestamp": datetime.now().isoformat(),
                    "session_id": st.session_state.session_id,
                    "pdf_name": st.session_state.active_pdf
                }

                # Append to chat history
                st.session_state.chat_history.append(chat_entry)
                
                # Save to persistent storage
                save_chat_history(st.session_state.chat_history)
                
                # Force a rerun to update the chat history and clear the form
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during retrieval/generation: {e}")
    
    with col2:
        st.markdown("---")
        st.subheader("📚 Previous Sessions")
        
        # Show previous sessions chat history
        previous_sessions_chats = [
            chat for chat in st.session_state.chat_history 
            if chat.get("session_id") != st.session_state.session_id
        ]
        
        if previous_sessions_chats:
            # Group by session and PDF
            sessions_data = {}
            for chat in previous_sessions_chats:
                session_id = chat.get("session_id", "unknown")
                pdf_name = chat.get("pdf_name", "unknown")
                
                if session_id not in sessions_data:
                    sessions_data[session_id] = {}
                if pdf_name not in sessions_data[session_id]:
                    sessions_data[session_id][pdf_name] = []
                    
                sessions_data[session_id][pdf_name].append(chat)
            
            # Display previous sessions in reverse chronological order
            sorted_sessions = sorted(sessions_data.items(), key=lambda x: 
                max([chat.get('timestamp', '') for chat in sum(x[1].values(), [])]), 
                reverse=True
            )
            
            for session_id, pdfs_data in sorted_sessions[:5]:  # Show only last 5 sessions
                session_display_id = session_id[:6] + "..."
                
                # Get latest timestamp for this session
                latest_timestamp = max([
                    chat.get('timestamp', '') 
                    for chats in pdfs_data.values() 
                    for chat in chats
                ])
                
                # Format date
                try:
                    date_str = datetime.fromisoformat(latest_timestamp).strftime("%b %d, %H:%M")
                except:
                    date_str = "Unknown date"
                
                with st.expander(f"🕒 {date_str} ({session_display_id})", expanded=False):
                    for pdf_name, chats in pdfs_data.items():
                        st.markdown(f"**📄 {pdf_name}**")
                        # Show last 2 messages per PDF to save space
                        for chat in chats[-2:]:
                            st.markdown(f"**Q:** {chat['user'][:50]}{'...' if len(chat['user']) > 50 else ''}")
                            st.markdown(f"**A:** {chat['bot'][:70]}{'...' if len(chat['bot']) > 70 else ''}")
                        st.markdown("---")
        else:
            st.info("No previous sessions found.")

else:
    # When no PDF is processed, show full width previous sessions
    st.info("👈 Please upload and process PDF files to start chatting.")
    
    # Show previous sessions chat history
    previous_sessions_chats = [
        chat for chat in st.session_state.chat_history 
        if chat.get("session_id") != st.session_state.session_id
    ]
    
    if previous_sessions_chats:
        st.markdown("---")
        st.subheader("📚 Previous Sessions Chat History")
        
        # Group by session and PDF
        sessions_data = {}
        for chat in previous_sessions_chats:
            session_id = chat.get("session_id", "unknown")
            pdf_name = chat.get("pdf_name", "unknown")
            
            if session_id not in sessions_data:
                sessions_data[session_id] = {}
            if pdf_name not in sessions_data[session_id]:
                sessions_data[session_id][pdf_name] = []
                
            sessions_data[session_id][pdf_name].append(chat)
        
        # Display previous sessions in reverse chronological order
        sorted_sessions = sorted(sessions_data.items(), key=lambda x: 
            max([chat.get('timestamp', '') for chat in sum(x[1].values(), [])]), 
            reverse=True
        )
        
        for session_id, pdfs_data in sorted_sessions[:8]:  # Show more when no active PDF
            session_display_id = session_id[:6] + "..."
            
            # Get latest timestamp for this session
            latest_timestamp = max([
                chat.get('timestamp', '') 
                for chats in pdfs_data.values() 
                for chat in chats
            ])
            
            # Format date
            try:
                date_str = datetime.fromisoformat(latest_timestamp).strftime("%b %d, %H:%M")
            except:
                date_str = "Unknown date"
            
            with st.expander(f"🕒 Session: {date_str} ({session_display_id})", expanded=False):
                for pdf_name, chats in pdfs_data.items():
                    st.markdown(f"**📄 PDF:** {pdf_name}")
                    for chat in chats[-3:]:
                        st.markdown(f"**You:** {chat['user']}")
                        st.markdown(f"**AI:** {chat['bot']}")
                        st.markdown("---")

# ----------------------------
# Auto-cleanup when session ends
# ----------------------------
def cleanup_before_exit():
    """Clean up session data before exit"""
    if not st.session_state.get("session_cleanup_done", False):
        cleanup_current_session_data()
        cleanup_temp_files()

# Register cleanup function
import atexit
atexit.register(cleanup_before_exit)

# Also cleanup temp files on rerun
cleanup_temp_files()