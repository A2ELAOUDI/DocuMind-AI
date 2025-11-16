"""
Tests unitaires pour DocuMind AI
Ces tests vérifient que chaque composant fonctionne correctement
"""

import pytest
import os
import sys
from pathlib import Path

# Ajoute le répertoire parent au path pour importer les modules
sys.path.append(str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.utils import (
    is_supported_file,
    get_file_extension,
    sanitize_filename,
    count_tokens_approximate
)


# ============================================================================
# TESTS UTILS
# ============================================================================

def test_get_file_extension():
    """Test extraction d'extension de fichier"""
    assert get_file_extension("document.pdf") == ".pdf"
    assert get_file_extension("guide.MD") == ".md"  # Teste lowercase
    assert get_file_extension("file.txt") == ".txt"


def test_is_supported_file():
    """Test vérification des formats supportés"""
    assert is_supported_file("doc.pdf") == True
    assert is_supported_file("doc.txt") == True
    assert is_supported_file("doc.md") == True
    assert is_supported_file("doc.docx") == True
    assert is_supported_file("doc.exe") == False
    assert is_supported_file("doc.jpg") == False


def test_sanitize_filename():
    """Test nettoyage des noms de fichiers"""
    assert sanitize_filename("normal.pdf") == "normal.pdf"
    assert sanitize_filename("file:with:colons.txt") == "file_with_colons.txt"
    assert sanitize_filename("../../../etc/passwd") == ".._.._.._etc_passwd"


def test_count_tokens_approximate():
    """Test estimation du nombre de tokens"""
    text = "Hello world"  # 11 caractères
    tokens = count_tokens_approximate(text)
    assert tokens == 2  # 11 // 4 = 2


# ============================================================================
# TESTS DOCUMENT PROCESSOR
# ============================================================================

def test_document_processor_initialization():
    """Test initialisation du DocumentProcessor"""
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    assert processor.chunk_size == 500
    assert processor.chunk_overlap == 50
    assert processor.text_splitter is not None


def test_document_processor_split():
    """Test découpage de documents"""
    from langchain.schema import Document

    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    # Crée un document de test
    long_text = "A" * 300  # 300 caractères
    docs = [Document(page_content=long_text, metadata={"source": "test"})]

    # Découpe
    chunks = processor.split_documents(docs)

    # Vérifie qu'il y a plusieurs chunks
    assert len(chunks) > 1

    # Vérifie que les chunks ont la bonne taille (environ)
    for chunk in chunks:
        assert len(chunk.page_content) <= 120  # chunk_size + marge


# ============================================================================
# TESTS VECTOR STORE
# ============================================================================

@pytest.fixture
def temp_vector_store(tmp_path):
    """
    Fixture: crée un VectorStore temporaire pour les tests

    tmp_path est fourni par pytest, c'est un répertoire temporaire
    qui sera automatiquement nettoyé après le test
    """
    persist_dir = tmp_path / "test_vectordb"
    vs = VectorStore(persist_directory=str(persist_dir), use_ollama=False)
    return vs


def test_vector_store_initialization(temp_vector_store):
    """Test initialisation du VectorStore"""
    vs = temp_vector_store
    assert vs.persist_directory is not None
    assert vs.embeddings is not None


def test_vector_store_stats(temp_vector_store):
    """Test récupération des statistiques"""
    vs = temp_vector_store
    stats = vs.get_collection_stats()
    assert "status" in stats
    # Au départ, count devrait être 0
    assert stats.get("count", 0) == 0


# NOTE: Les tests suivants nécessitent une clé API OpenAI valide
# Ils sont marqués avec @pytest.mark.skipif pour être ignorés si pas de clé

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not found"
)
def test_vector_store_add_documents(temp_vector_store):
    """Test ajout de documents au VectorStore"""
    from langchain.schema import Document

    vs = temp_vector_store

    # Crée des documents de test
    docs = [
        Document(page_content="Python est un langage de programmation", metadata={"source": "doc1"}),
        Document(page_content="JavaScript est utilisé pour le web", metadata={"source": "doc2"}),
    ]

    # Ajoute les documents
    vs.add_documents(docs)

    # Vérifie les stats
    stats = vs.get_collection_stats()
    assert stats["count"] == 2


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not found"
)
def test_vector_store_similarity_search(temp_vector_store):
    """Test recherche de similarité"""
    from langchain.schema import Document

    vs = temp_vector_store

    # Indexe des documents
    docs = [
        Document(page_content="Python est un langage de programmation", metadata={"source": "doc1"}),
        Document(page_content="Le chat est un animal domestique", metadata={"source": "doc2"}),
    ]
    vs.add_documents(docs)

    # Recherche
    results = vs.similarity_search("langage de programmation", k=2)

    # Vérifie qu'on a des résultats
    assert len(results) > 0

    # Le premier résultat devrait être le doc1 (plus pertinent)
    assert "Python" in results[0].page_content


# ============================================================================
# TESTS RAG ENGINE
# ============================================================================

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not found"
)
def test_rag_engine_initialization(temp_vector_store):
    """Test initialisation du RAG Engine"""
    vs = temp_vector_store

    # Ajoute un document pour initialiser le vectorstore
    from langchain.schema import Document
    vs.add_documents([
        Document(page_content="Test content", metadata={"source": "test"})
    ])

    # Crée le RAG engine
    rag = RAGEngine(vector_store=vs, use_ollama=False)

    assert rag.llm is not None
    assert rag.vector_store is not None


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not found"
)
def test_rag_engine_query(temp_vector_store):
    """Test query du RAG Engine"""
    from langchain.schema import Document

    vs = temp_vector_store

    # Indexe des documents
    docs = [
        Document(
            page_content="Python est un langage de programmation créé par Guido van Rossum",
            metadata={"source": "python_guide", "page": 1}
        ),
        Document(
            page_content="Pour installer Python, téléchargez-le depuis python.org",
            metadata={"source": "python_guide", "page": 2}
        ),
    ]
    vs.add_documents(docs)

    # Crée le RAG engine
    rag = RAGEngine(vector_store=vs, use_ollama=False)

    # Pose une question
    response = rag.query("Qui a créé Python?")

    # Vérifie la réponse
    assert "answer" in response
    assert "sources" in response
    assert len(response["answer"]) > 0

    # La réponse devrait mentionner Guido (si le LLM fonctionne bien)
    # Note: On ne peut pas garantir 100% que le LLM dira exactement "Guido"
    # mais on peut vérifier qu'il y a une réponse
    assert isinstance(response["answer"], str)


# ============================================================================
# TESTS D'INTÉGRATION
# ============================================================================

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not found"
)
def test_full_pipeline(tmp_path):
    """
    Test du pipeline complet: DocumentProcessor → VectorStore → RAGEngine

    C'est un test d'intégration qui vérifie que tous les composants
    fonctionnent bien ensemble
    """
    from langchain.schema import Document

    # Setup
    persist_dir = tmp_path / "test_db"
    vs = VectorStore(persist_directory=str(persist_dir))

    # Crée un document de test
    test_content = """
    Python est un langage de programmation interprété.
    Il a été créé par Guido van Rossum et publié en 1991.
    Python est utilisé pour le développement web, l'analyse de données, l'IA, etc.
    Pour installer Python, rendez-vous sur python.org.
    """

    docs = [Document(page_content=test_content, metadata={"source": "python_doc"})]

    # Découpe avec DocumentProcessor
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    chunks = processor.split_documents(docs)

    # Vérifie qu'on a bien des chunks
    assert len(chunks) > 0

    # Indexe dans VectorStore
    vs.add_documents(chunks)

    # Vérifie l'indexation
    stats = vs.get_collection_stats()
    assert stats["count"] > 0

    # Crée le RAG Engine
    rag = RAGEngine(vector_store=vs)

    # Pose une question
    response = rag.query("Qui a créé Python?")

    # Vérifie la structure de la réponse
    assert "answer" in response
    assert "sources" in response
    assert "metadata" in response

    # Vérifie qu'on a des sources
    assert len(response["sources"]) > 0


# ============================================================================
# TESTS API (optionnels)
# ============================================================================

def test_api_imports():
    """Test que les imports de l'API fonctionnent"""
    try:
        from api.main import app
        assert app is not None
    except ImportError as e:
        pytest.fail(f"Failed to import API: {e}")


# ============================================================================
# COMMENT LANCER LES TESTS
# ============================================================================

"""
Pour lancer les tests:

1. Install pytest:
   pip install pytest pytest-asyncio

2. Lancer tous les tests:
   pytest tests/

3. Lancer avec verbose:
   pytest tests/ -v

4. Lancer un test spécifique:
   pytest tests/test_rag.py::test_get_file_extension

5. Lancer avec coverage:
   pip install pytest-cov
   pytest --cov=src tests/

6. Skip les tests nécessitant OpenAI:
   pytest tests/ -v -m "not skipif"

7. Voir les print statements:
   pytest tests/ -v -s
"""

if __name__ == "__main__":
    # Permet de lancer les tests avec: python tests/test_rag.py
    pytest.main([__file__, "-v"])
