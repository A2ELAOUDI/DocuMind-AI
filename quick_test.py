"""
Script de Test Rapide - DocuMind AI
Lance ce script pour tester rapidement le système RAG

Usage:
    python quick_test.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Charge les variables d'environnement
load_dotenv()

# Imports du projet
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.document_processor import DocumentProcessor

# Couleurs pour le terminal (optionnel)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Affiche un header coloré"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}\n")

def print_success(text):
    """Affiche un message de succès"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    """Affiche un message d'erreur"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Affiche un message d'info"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def print_warning(text):
    """Affiche un warning"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def check_api_key():
    """Vérifie que la clé API est configurée"""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "your_openai_api_key_here":
        print_error("Clé API OpenAI non configurée!")
        print_info("Édite le fichier .env et ajoute ta clé:")
        print_info("OPENAI_API_KEY=sk-your-key-here")
        return False

    print_success("Clé API OpenAI trouvée")
    return True

def test_imports():
    """Teste que tous les imports fonctionnent"""
    print_header("TEST 1: Vérification des Imports")

    try:
        import langchain
        print_success(f"LangChain importé (version: {langchain.__version__})")
    except ImportError as e:
        print_error(f"LangChain import failed: {e}")
        return False

    try:
        import chromadb
        print_success("ChromaDB importé")
    except ImportError as e:
        print_error(f"ChromaDB import failed: {e}")
        return False

    try:
        import openai
        print_success(f"OpenAI importé (version: {openai.__version__})")
    except ImportError as e:
        print_error(f"OpenAI import failed: {e}")
        return False

    return True

def test_document_processor():
    """Teste le DocumentProcessor"""
    print_header("TEST 2: Document Processor")

    try:
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        print_success("DocumentProcessor initialisé")

        # Teste avec le document d'exemple
        example_file = "data/documents/example_python_guide.md"

        if os.path.exists(example_file):
            print_info(f"Traitement de {example_file}...")
            chunks = processor.process_file(example_file)
            print_success(f"Document traité: {len(chunks)} chunks créés")

            # Affiche le premier chunk (aperçu)
            print_info("Premier chunk (aperçu):")
            print(f"  {chunks[0].page_content[:150]}...")

            return True, chunks
        else:
            print_warning(f"Fichier d'exemple non trouvé: {example_file}")
            print_info("Crée un fichier test.txt dans data/documents/")
            return True, []

    except Exception as e:
        print_error(f"Erreur DocumentProcessor: {e}")
        return False, []

def test_vector_store(chunks):
    """Teste le VectorStore"""
    print_header("TEST 3: Vector Store")

    if not chunks:
        print_warning("Pas de chunks à indexer, skip")
        return True, None

    try:
        vs = VectorStore(persist_directory="./data/test_vectordb")
        print_success("VectorStore initialisé")

        print_info("Indexation des chunks...")
        vs.add_documents(chunks)
        print_success("Documents indexés!")

        # Stats
        stats = vs.get_collection_stats()
        print_info(f"Stats: {stats['count']} embeddings dans la DB")

        # Test de recherche
        print_info("Test de recherche...")
        results = vs.similarity_search("Comment installer Python?", k=2)
        print_success(f"Recherche OK: {len(results)} résultats trouvés")

        # Affiche le premier résultat
        if results:
            print_info("Meilleur résultat:")
            print(f"  {results[0].page_content[:150]}...")

        return True, vs

    except Exception as e:
        print_error(f"Erreur VectorStore: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_rag_engine(vector_store):
    """Teste le RAG Engine"""
    print_header("TEST 4: RAG Engine")

    if vector_store is None:
        print_warning("VectorStore non disponible, skip")
        return True

    try:
        rag = RAGEngine(vector_store=vector_store, use_ollama=False)
        print_success("RAG Engine initialisé")

        # Questions de test
        questions = [
            "Comment installer Python sur Windows?",
            "Qu'est-ce qu'une list comprehension en Python?",
            "Quelles sont les bonnes pratiques PEP 8?"
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n{Colors.BOLD}Question {i}:{Colors.END} {question}")

            try:
                response = rag.query(question)

                print(f"{Colors.GREEN}Réponse:{Colors.END}")
                print(f"  {response['answer'][:300]}...")

                print(f"\n{Colors.BLUE}Sources:{Colors.END}")
                for j, source in enumerate(response['sources'][:2], 1):
                    print(f"  [{j}] {source['source_file']} (page {source['page']})")

                print_success(f"Question {i} traitée avec succès")

            except Exception as e:
                print_error(f"Erreur pour question {i}: {e}")

        return True

    except Exception as e:
        print_error(f"Erreur RAG Engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup():
    """Nettoie les fichiers de test"""
    print_header("Nettoyage")

    import shutil

    test_db = "./data/test_vectordb"
    if os.path.exists(test_db):
        try:
            shutil.rmtree(test_db)
            print_success("Base de test supprimée")
        except Exception as e:
            print_warning(f"Impossible de supprimer {test_db}: {e}")

def main():
    """Fonction principale"""
    print_header("DocuMind AI - Test Rapide")

    # Vérifications préalables
    if not check_api_key():
        print_error("\nTest annulé: Configure ta clé API d'abord!")
        return

    print()

    # Tests
    all_passed = True

    # Test 1: Imports
    if not test_imports():
        all_passed = False
        print_error("\nTest interrompu: Problème d'imports")
        return

    # Test 2: Document Processor
    success, chunks = test_document_processor()
    if not success:
        all_passed = False

    # Test 3: Vector Store
    success, vs = test_vector_store(chunks)
    if not success:
        all_passed = False

    # Test 4: RAG Engine
    if vs is not None:
        success = test_rag_engine(vs)
        if not success:
            all_passed = False

    # Nettoyage
    cleanup()

    # Résultat final
    print_header("Résultat")

    if all_passed:
        print_success("✅ TOUS LES TESTS SONT PASSÉS!")
        print_info("\nTu peux maintenant:")
        print_info("  1. Lancer l'interface: streamlit run ui/streamlit_app.py")
        print_info("  2. Lancer l'API: uvicorn api.main:app --reload")
        print_info("  3. Lire le LEARNING_GUIDE.md pour tout comprendre!")
    else:
        print_error("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print_info("\nVérifie:")
        print_info("  1. Que toutes les dépendances sont installées (pip install -r requirements.txt)")
        print_info("  2. Que ta clé API est valide")
        print_info("  3. Les messages d'erreur ci-dessus")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\nTest interrompu par l'utilisateur")
    except Exception as e:
        print_error(f"\n\nErreur inattendue: {e}")
        import traceback
        traceback.print_exc()
