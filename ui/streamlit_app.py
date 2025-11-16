"""
Streamlit UI - Interface utilisateur pour DocuMind AI
Application web interactive pour interagir avec le système RAG
"""

import streamlit as st
import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Import des modules du projet
sys.path.append(str(Path(__file__).parent.parent))

from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.document_processor import DocumentProcessor
from src.utils import is_supported_file

# Chargement des variables d'environnement
load_dotenv()


# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",  # Layout large pour utiliser tout l'écran
    initial_sidebar_state="expanded"  # Sidebar ouverte par défaut
)


# ============================================================================
# GESTION DE L'ÉTAT (Session State)
# ============================================================================

# Streamlit recharge le script à chaque interaction
# Session State permet de persister des variables entre les reloads

def initialize_session_state():
    """
    Initialise les variables de session si elles n'existent pas

    Session State:
        - Similaire à des variables globales
        - Persistent entre les interactions utilisateur
        - Crucial pour ne pas réinitialiser le RAG à chaque clic
    """
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False

    if "current_stats" not in st.session_state:
        st.session_state.current_stats = {"count": 0}


def initialize_rag_system():
    """
    Initialise le système RAG (une seule fois)

    Cette fonction est appelée au premier chargement de l'app
    ou quand l'utilisateur clique sur "Initialize System"
    """
    try:
        with st.spinner("Initializing RAG system..."):
            # Configuration
            use_ollama = os.getenv("LLM_PROVIDER", "openai").lower() == "ollama"
            persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/vectordb")

            # Vector Store
            st.session_state.vector_store = VectorStore(
                persist_directory=persist_dir,
                use_ollama=use_ollama
            )

            # Tentative de chargement d'une base existante
            loaded = st.session_state.vector_store.load_existing()

            # RAG Engine
            model_name = os.getenv("OLLAMA_MODEL", "llama2") if use_ollama else "gpt-3.5-turbo"
            temperature = float(os.getenv("TEMPERATURE", "0.7"))
            max_tokens = int(os.getenv("MAX_TOKENS", "500"))

            st.session_state.rag_engine = RAGEngine(
                vector_store=st.session_state.vector_store,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                use_ollama=use_ollama
            )

            # Mise à jour des stats
            st.session_state.current_stats = st.session_state.vector_store.get_collection_stats()
            st.session_state.documents_loaded = loaded

            if loaded:
                st.success(f"✅ System initialized! Found {st.session_state.current_stats['count']} existing embeddings.")
            else:
                st.info("ℹ️ System initialized. Upload documents to get started!")

    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")


# ============================================================================
# COMPOSANTS DE L'INTERFACE
# ============================================================================

def render_sidebar():
    """
    Affiche la barre latérale avec les contrôles et stats

    Sidebar:
        - Configuration
        - Upload de documents
        - Statistiques
        - Contrôles système
    """
    with st.sidebar:
        st.title("🧠 DocuMind AI")
        st.markdown("---")

        # Section: System Status
        st.subheader("System Status")

        if st.session_state.rag_engine is None:
            st.warning("⚠️ System not initialized")

            if st.button("🚀 Initialize System", use_container_width=True):
                initialize_rag_system()
        else:
            st.success("✅ System Ready")

            # Affichage des statistiques
            stats = st.session_state.current_stats
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Documents", stats.get("count", 0))

            with col2:
                provider = os.getenv("LLM_PROVIDER", "openai").upper()
                st.metric("LLM", provider)

        st.markdown("---")

        # Section: Upload Documents
        st.subheader("📁 Upload Documents")

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, MD, DOCX"
        )

        if uploaded_files and st.button("📤 Upload & Index", use_container_width=True):
            process_uploads(uploaded_files)

        st.markdown("---")

        # Section: Settings
        with st.expander("⚙️ Settings"):
            st.text(f"Chunk Size: {os.getenv('CHUNK_SIZE', '1000')}")
            st.text(f"Chunk Overlap: {os.getenv('CHUNK_OVERLAP', '200')}")
            st.text(f"Temperature: {os.getenv('TEMPERATURE', '0.7')}")
            st.text(f"Max Tokens: {os.getenv('MAX_TOKENS', '500')}")

        # Section: Danger Zone
        with st.expander("⚠️ Danger Zone"):
            if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
                reset_knowledge_base()


def process_uploads(uploaded_files: List):
    """
    Traite les fichiers uploadés par l'utilisateur

    Args:
        uploaded_files: Liste de fichiers Streamlit UploadedFile

    Process:
        1. Valide les fichiers
        2. Sauvegarde temporaire
        3. Indexation
        4. Mise à jour de l'interface
    """
    if st.session_state.rag_engine is None:
        st.error("Please initialize the system first!")
        return

    temp_dir = Path("./data/temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_paths = []

    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Sauvegarde des fichiers
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Saving {uploaded_file.name}...")

            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            file_paths.append(str(file_path))

            progress_bar.progress((i + 1) / len(uploaded_files) / 2)

        # Indexation
        status_text.text("Indexing documents...")

        stats = st.session_state.rag_engine.add_documents_to_knowledge_base(file_paths)

        # Mise à jour des stats
        st.session_state.current_stats = st.session_state.vector_store.get_collection_stats()
        st.session_state.documents_loaded = True

        progress_bar.progress(1.0)

        # Affichage des résultats
        st.success(f"""
        ✅ Upload successful!
        - Files processed: {stats['processed_files']}/{stats['total_files']}
        - Chunks created: {stats['total_chunks']}
        """)

        if stats['failed_files'] > 0:
            st.warning(f"⚠️ {stats['failed_files']} file(s) failed to process")

        # Nettoyage
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except:
                pass

    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")

    finally:
        progress_bar.empty()
        status_text.empty()


def reset_knowledge_base():
    """
    Supprime complètement la base de connaissances
    """
    if st.session_state.vector_store is None:
        st.warning("No vector store to reset")
        return

    try:
        st.session_state.vector_store.delete_collection()
        st.session_state.current_stats = {"count": 0}
        st.session_state.documents_loaded = False
        st.session_state.chat_history = []

        st.success("✅ Knowledge base reset successfully!")
        st.rerun()  # Recharge l'app

    except Exception as e:
        st.error(f"❌ Error resetting: {str(e)}")


def render_chat_interface():
    """
    Affiche l'interface de chat principale

    Interface:
        - Historique des messages
        - Input utilisateur
        - Affichage des réponses avec sources
    """
    st.title("💬 Chat with your Documents")

    # Zone de chat (historique)
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            # Message utilisateur
            with st.chat_message("user"):
                st.write(message["question"])

            # Réponse de l'assistant
            with st.chat_message("assistant"):
                st.write(message["answer"])

                # Affichage des sources
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.markdown(f"""
                            **{source['source_file']}** (Page {source['page']})
                            > {source['content']}
                            """)

    # Input utilisateur
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        process_question(user_question)


def process_question(question: str):
    """
    Traite une question de l'utilisateur

    Args:
        question: Question posée par l'utilisateur

    Process:
        1. Validation
        2. Appel du RAG engine
        3. Affichage de la réponse
        4. Mise à jour de l'historique
    """
    if st.session_state.rag_engine is None:
        st.error("Please initialize the system first!")
        return

    if not st.session_state.documents_loaded:
        st.warning("Please upload some documents first!")
        return

    try:
        # Affichage immédiat de la question
        with st.chat_message("user"):
            st.write(question)

        # Génération de la réponse
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_engine.query(question)

            # Affichage de la réponse
            st.write(response["answer"])

            # Affichage des sources
            if response["sources"]:
                with st.expander("📚 Sources"):
                    for source in response["sources"]:
                        st.markdown(f"""
                        **{source['source_file']}** (Page {source['page']})
                        > {source['content']}
                        """)

        # Ajout à l'historique
        st.session_state.chat_history.append({
            "question": question,
            "answer": response["answer"],
            "sources": response["sources"]
        })

    except Exception as e:
        st.error(f"❌ Error processing question: {str(e)}")


def render_welcome_screen():
    """
    Affiche l'écran de bienvenue quand aucun document n'est chargé
    """
    st.title("🧠 Welcome to DocuMind AI")

    st.markdown("""
    ## Smart Documentation Assistant

    DocuMind AI uses **Retrieval Augmented Generation (RAG)** to answer questions about your documents.

    ### How it works:
    1. **Upload** your documents (PDF, TXT, MD, DOCX)
    2. **Ask** questions in natural language
    3. **Get** accurate answers with sources

    ### Features:
    - 📚 Support for multiple document formats
    - 🔍 Semantic search (not just keywords)
    - 🎯 Answers with source citations
    - 💬 Chat interface with history
    - 🚀 Fast and accurate

    ### Getting Started:
    1. Click "Initialize System" in the sidebar
    2. Upload your documents
    3. Start asking questions!
    """)

    # Statistiques techniques
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Vector Database**
        ChromaDB
        Local & Fast
        """)

    with col2:
        st.info("""
        **LLM**
        GPT-3.5 / Ollama
        Configurable
        """)

    with col3:
        st.info("""
        **Framework**
        LangChain
        Production-Ready
        """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """
    Fonction principale de l'application

    Flow:
        1. Initialisation du session state
        2. Affichage de la sidebar
        3. Affichage du contenu principal
    """
    # Initialisation
    initialize_session_state()

    # Sidebar (toujours visible)
    render_sidebar()

    # Contenu principal
    if st.session_state.documents_loaded:
        # Interface de chat si des documents sont chargés
        render_chat_interface()
    else:
        # Écran de bienvenue sinon
        render_welcome_screen()

    # Footer
    st.markdown("---")
    st.caption("DocuMind AI v1.0 - Built with LangChain, ChromaDB & Streamlit")


# Point d'entrée
if __name__ == "__main__":
    main()

# Pour lancer l'app:
# streamlit run ui/streamlit_app.py
