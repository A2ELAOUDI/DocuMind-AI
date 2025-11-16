"""
FastAPI Application - API REST pour DocuMind AI
Cette API expose le système RAG via des endpoints HTTP
"""

import logging
import os
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import des modules du projet
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.document_processor import DocumentProcessor
from src.utils import is_supported_file, sanitize_filename

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="DocuMind AI API",
    description="API REST pour le système RAG de documentation intelligente",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI automatique
    redoc_url="/redoc"  # Documentation ReDoc
)

# Configuration CORS (permet les requêtes depuis le frontend)
# CORS = Cross-Origin Resource Sharing
# Nécessaire si ton UI et ton API sont sur des ports différents
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod, spécifie les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# MODÈLES PYDANTIC (Schémas de données)
# ============================================================================

class QueryRequest(BaseModel):
    """
    Schéma pour les requêtes de question

    Pydantic:
        - Validation automatique des données
        - Documentation auto dans Swagger
        - Type hints pour l'auto-complétion
    """
    question: str = Field(
        ...,  # ... = requis
        description="Question à poser au système RAG",
        min_length=3,
        max_length=500,
        example="Comment installer Python sur Windows?"
    )
    k: int = Field(
        default=4,
        description="Nombre de documents à récupérer",
        ge=1,  # Greater or equal
        le=10  # Less or equal
    )


class QueryResponse(BaseModel):
    """
    Schéma pour les réponses du système
    """
    answer: str = Field(description="Réponse générée par le LLM")
    sources: List[dict] = Field(description="Sources utilisées pour la réponse")
    metadata: dict = Field(description="Métadonnées de la requête")


class UploadResponse(BaseModel):
    """
    Schéma pour les réponses d'upload de fichiers
    """
    message: str
    stats: dict
    uploaded_files: List[str]


class HealthResponse(BaseModel):
    """
    Schéma pour le health check
    """
    status: str
    vector_store_status: dict
    environment: str


# ============================================================================
# INITIALISATION DES COMPOSANTS
# ============================================================================

# Variables globales pour le RAG engine
# En production, utiliser un pattern Singleton ou Dependency Injection
vector_store: Optional[VectorStore] = None
rag_engine: Optional[RAGEngine] = None


def initialize_rag_system():
    """
    Initialise le système RAG au démarrage de l'API

    Pourquoi au démarrage ?
        - Évite de réinitialiser à chaque requête
        - Plus rapide pour les requêtes ultérieures
        - Charge la DB vectorielle une seule fois
    """
    global vector_store, rag_engine

    try:
        logger.info("Initializing RAG system...")

        # Configuration depuis les variables d'environnement
        use_ollama = os.getenv("LLM_PROVIDER", "openai").lower() == "ollama"
        persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/vectordb")

        # Initialisation du vector store
        vector_store = VectorStore(
            persist_directory=persist_dir,
            use_ollama=use_ollama
        )

        # Tentative de chargement d'une DB existante
        loaded = vector_store.load_existing()
        if loaded:
            logger.info("✓ Existing vector store loaded")
        else:
            logger.info("! No existing vector store found - will be created on first upload")

        # Initialisation du RAG engine
        model_name = os.getenv("OLLAMA_MODEL", "llama2") if use_ollama else "gpt-3.5-turbo"
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
        max_tokens = int(os.getenv("MAX_TOKENS", "500"))

        rag_engine = RAGEngine(
            vector_store=vector_store,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            use_ollama=use_ollama
        )

        logger.info("✓ RAG system initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        # On laisse quand même l'API démarrer (les endpoints retourneront des erreurs claires)


# Event handler: s'exécute au démarrage de l'application
@app.on_event("startup")
async def startup_event():
    """
    Événement déclenché au démarrage de FastAPI
    """
    logger.info("Starting DocuMind AI API...")
    initialize_rag_system()
    logger.info("API ready to accept requests")


# ============================================================================
# ENDPOINTS (Routes de l'API)
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """
    Endpoint racine - Page d'accueil de l'API

    Tags:
        - Organisent les endpoints dans la doc Swagger
    """
    return {
        "message": "DocuMind AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check - Vérifie que l'API fonctionne correctement

    Utile pour:
        - Monitoring (Kubernetes, Docker)
        - CI/CD pipelines
        - Load balancers
    """
    if rag_engine is None or vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )

    return {
        "status": "healthy",
        "vector_store_status": vector_store.get_collection_stats(),
        "environment": os.getenv("LLM_PROVIDER", "openai")
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_documents(request: QueryRequest):
    """
    Pose une question au système RAG

    Process:
        1. Reçoit la question
        2. Recherche dans la base vectorielle
        3. Génère une réponse avec le LLM
        4. Retourne réponse + sources

    Exemple de requête:
        POST /query
        {
            "question": "Comment installer Python?",
            "k": 4
        }

    Exemple de réponse:
        {
            "answer": "Pour installer Python, téléchargez...",
            "sources": [...],
            "metadata": {...}
        }
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized. Please upload documents first."
        )

    try:
        logger.info(f"Received query: {request.question}")

        # Appel du RAG engine
        response = rag_engine.query(request.question)

        return QueryResponse(
            answer=response["answer"],
            sources=response["sources"],
            metadata=response["metadata"]
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload et indexation de documents

    Args:
        files: Liste de fichiers uploadés (multipart/form-data)

    Process:
        1. Reçoit les fichiers
        2. Sauvegarde temporairement
        3. Traite avec DocumentProcessor
        4. Indexe dans la base vectorielle
        5. Supprime les fichiers temporaires

    Formats supportés: PDF, TXT, MD, DOCX

    Exemple avec curl:
        curl -X POST "http://localhost:8000/upload" \
             -F "files=@document1.pdf" \
             -F "files=@document2.md"
    """
    if rag_engine is None or vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )

    uploaded_files = []
    temp_dir = Path("./data/temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Sauvegarde des fichiers uploadés
        file_paths = []

        for file in files:
            # Validation du type de fichier
            if not is_supported_file(file.filename):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file.filename}"
                )

            # Sauvegarde temporaire
            safe_filename = sanitize_filename(file.filename)
            file_path = temp_dir / safe_filename

            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            file_paths.append(str(file_path))
            uploaded_files.append(safe_filename)

        logger.info(f"Uploaded {len(file_paths)} file(s)")

        # Indexation des documents
        stats = rag_engine.add_documents_to_knowledge_base(file_paths)

        # Nettoyage des fichiers temporaires
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")

        return UploadResponse(
            message="Documents uploaded and indexed successfully",
            stats=stats,
            uploaded_files=uploaded_files
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading documents: {str(e)}"
        )


@app.get("/stats", tags=["Documents"])
async def get_statistics():
    """
    Retourne des statistiques sur la base de connaissances

    Informations:
        - Nombre de documents indexés
        - Taille de la base vectorielle
        - Configuration du système
    """
    if vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized"
        )

    return {
        "vector_store": vector_store.get_collection_stats(),
        "configuration": {
            "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
            "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200"))
        }
    }


@app.delete("/reset", tags=["Documents"])
async def reset_knowledge_base():
    """
    Supprime complètement la base de connaissances

    ⚠️ ATTENTION: Opération irréversible !

    Utile pour:
        - Reset complet du système
        - Tests
        - Changement de configuration
    """
    if vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized"
        )

    try:
        vector_store.delete_collection()
        logger.info("Knowledge base reset successfully")

        return {
            "message": "Knowledge base deleted successfully",
            "status": "empty"
        }

    except Exception as e:
        logger.error(f"Error resetting knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resetting knowledge base: {str(e)}"
        )


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Uvicorn = serveur ASGI pour FastAPI
    # ASGI = Async Server Gateway Interface (version async de WSGI)

    uvicorn.run(
        "main:app",  # Module:app
        host="0.0.0.0",  # Écoute sur toutes les interfaces réseau
        port=8000,  # Port par défaut
        reload=True,  # Auto-reload en dev (détecte les changements de code)
        log_level="info"
    )

    # Pour lancer:
    # python api/main.py
    # ou
    # uvicorn api.main:app --reload
