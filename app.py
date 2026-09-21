import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

from src.ollama_chain import OllamaChain, OllamaRAGChain
from src.pdf_handler import extract_pdf
from src.vqa import answer_visual_question
from src.audio_processor import AudioProcessor
audio_processor = AudioProcessor()

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# --- FIXED: Load Chain with Session State ---
def load_chain(_chat_memory):
    # Agar PDF mode ON hai AUR file uploaded hai
    if st.session_state.get('pdf_chat') and st.session_state.get('uploaded_file'):
        # Agar purani RAG chain maujood nahi ya knowledge change hui hai
        if 'rag_chain' not in st.session_state or st.session_state.get('knowledge_change'):
            try:
                st.session_state.rag_chain = OllamaRAGChain(_chat_memory, st.session_state.uploaded_file)
                st.session_state.knowledge_change = False
            except Exception as e:
                st.error(f"RAG Error: {e}")
                return OllamaChain(_chat_memory)
        return st.session_state.rag_chain
    
    # Default simple chat
    return OllamaChain(_chat_memory)

def file_uploader_change():
    # Jab bhi file badle, purani chain ko delete kar dein taake nayi file index ho
    if 'rag_chain' in st.session_state:
        del st.session_state.rag_chain
    st.session_state.knowledge_change = True
    if st.session_state.uploaded_file:
        st.session_state.pdf_chat = True
    clear_cache()

def toggle_pdf_chat_change():
    clear_cache()
    # Mode switch karne par knowledge base ko refresh karne ka signal dain
    st.session_state.knowledge_change = True

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def clear_cache():
    st.cache_resource.clear()

def initial_session_state():
    if 'send_input' not in st.session_state:
        st.session_state.send_input = False
    if 'knowledge_change' not in st.session_state:
        st.session_state.knowledge_change = False
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
    if 'pdf_chat' not in st.session_state:
        st.session_state.pdf_chat = False
    os.makedirs('./.cache/temp_files', exist_ok=True)

def main():
    st.set_page_config(page_title="Multimodal RAG", page_icon="🤖", layout="wide")
    st.title('🤖 Multimodal RAG')
    st.markdown("Welcome to the **Multimodal RAG** application. Upload PDFs, images, or audio and chat with your AI assistant.")
    
    initial_session_state()
    chat_container = st.container()

    # Sidebar setup
    st.sidebar.title("🛠️ Settings & Uploads")
    
    # Status indicator in sidebar
    if st.session_state.pdf_chat and st.session_state.uploaded_file:
        st.sidebar.success("✅ RAG Mode: Active (Using PDF)")
    else:
        st.sidebar.info("💬 Chat Mode: Simple")

    st.sidebar.toggle('PDF Chat Mode', key='pdf_chat', on_change=toggle_pdf_chat_change)
    
    with st.sidebar.expander("📄 Document Upload (PDF)", expanded=True):
        st.file_uploader('Upload your PDF files',
                                type='pdf',
                                accept_multiple_files=True,
                                key='uploaded_file',
                                on_change=file_uploader_change)

    with st.sidebar.expander("🖼️ & 🎵 Media Upload"):
        uploaded_image = st.file_uploader('Upload Images', type=['jpg', 'jpeg', 'png'], key='uploaded_image')
        uploaded_audio = st.file_uploader('Upload Audio', type=['wav', 'mp3'], key='uploaded_audio')

    chat_history = StreamlitChatMessageHistory(key='history')

    # Load appropriate chain (Ab ye function zyada stable hai)
    llm_chain = load_chain(chat_history)

    # Chat interface display
    with chat_container:
        for msg in chat_history.messages:
            st.chat_message(msg.type).write(msg.content)

    # User Input area
    user_query = st.chat_input('Message Multimodal RAG...')

    # Logic to process input
    if user_query:
        with chat_container:
            st.chat_message('user').write(user_query)
            
            with st.spinner("Thinking..."):
                if uploaded_image:
                    image_path = os.path.join('./.cache/temp_files', uploaded_image.name)
                    with open(image_path, 'wb') as f:
                        f.write(uploaded_image.getvalue())
                    llm_response = answer_visual_question(image_path, user_query)
                
                elif uploaded_audio:
                    audio_path = os.path.join('./.cache/temp_files', uploaded_audio.name)
                    with open(audio_path, 'wb') as f:
                        f.write(uploaded_audio.getvalue())
                    transcribed_text = audio_processor.audio_to_text(audio_path)
                    st.info(f"Transcribed: {transcribed_text}")
                    llm_response = llm_chain.run(user_input=transcribed_text)
                
                else:
                    llm_response = llm_chain.run(user_input=user_query)
            
            st.chat_message('ai').write(llm_response)

            # Audio Output logic
            try:
                audio_file = audio_processor.text_to_speech(llm_response)
                if audio_file and os.path.exists(audio_file):
                    with open(audio_file, 'rb') as f:
                        st.audio(f.read(), format='audio/mp3')
            except Exception:
                pass

if __name__ == '__main__':
    main()