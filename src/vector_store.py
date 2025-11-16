"""
Vector Store - Gestion de la base de données vectorielle avec ChromaDB
Ce module gère le stockage et la recherche des embeddings (représentations vectorielles)
"""

import logging
from typing import List, Optional
import os

# LangChain & ChromaDB imports
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from src.utils import ensure_directory_exists

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Classe qui gère la base de données vectorielle

    C'est quoi une base vectorielle ?
        - Stocke des "embeddings" (vecteurs numériques représentant le sens du texte)
        - Permet de chercher par similarité sémantique (pas juste mot-clé)
        - Exemple: "comment cuisiner des pâtes" trouvera aussi "recette de pasta"

    Comment ça marche ?
        1. Texte → Embedding model → Vecteur [0.2, -0.5, 0.8, ...]
        2. Stockage dans ChromaDB
        3. Recherche: Question → Vecteur → Trouver les vecteurs similaires → Documents

    Pourquoi ChromaDB ?
        - Open source et gratuit
        - Fonctionne en local (pas besoin de serveur)
        - Facile à utiliser
        - Parfait pour les POC et projets perso
    """

    def __init__(
        self,
        persist_directory: str = "./data/vectordb",
        embedding_model: str = "text-embedding-ada-002",
        collection_name: str = "documind",
        use_ollama: bool = False
    ):
        """
        Initialise le vector store

        Args:
            persist_directory: Où sauvegarder la DB sur le disque
            embedding_model: Modèle pour créer les embeddings
            collection_name: Nom de la collection ChromaDB
            use_ollama: Si True, utilise Ollama (local, gratuit) au lieu d'OpenAI

        Les embeddings, c'est quoi ?
            - Représentation numérique du sens d'un texte
            - Vecteur de nombres (ex: 1536 dimensions pour text-embedding-ada-002)
            - Textes similaires → vecteurs proches dans l'espace vectoriel

        OpenAI vs Ollama ?
            - OpenAI: Meilleure qualité, payant (~$0.0001/1K tokens)
            - Ollama: Gratuit, local, qualité un peu moindre mais suffisante
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.use_ollama = use_ollama

        # Création du répertoire si nécessaire
        ensure_directory_exists(persist_directory)

        # Initialisation du modèle d'embeddings
        if use_ollama:
            logger.info("Using Ollama embeddings (local, free)")
            self.embeddings = OllamaEmbeddings(
                model="llama2",  # ou "mistral", "codellama", etc.
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
        else:
            logger.info(f"Using OpenAI embeddings: {embedding_model}")
            # text-embedding-ada-002: Le modèle d'embedding d'OpenAI
            # 1536 dimensions, très performant
            api_key = os.getenv("OPENAI_API_KEY")
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=api_key
            )

        # Initialisation de ChromaDB (sera créée au premier ajout)
        self.vectorstore: Optional[Chroma] = None

        logger.info(f"VectorStore initialized (persist_dir: {persist_directory})")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Ajoute des documents à la base vectorielle

        Args:
            documents: Liste de chunks (Document LangChain)

        Process détaillé:
            1. Pour chaque document:
               - Extrait le texte (page_content)
               - Crée un embedding via l'API (OpenAI ou Ollama)
               - Stocke: (embedding, texte, métadonnées) dans ChromaDB
            2. Sauvegarde sur disque (persist)

        Coût (si OpenAI):
            - text-embedding-ada-002: $0.0001 / 1K tokens
            - Exemple: 100 chunks de 500 tokens = 50K tokens = $0.005
        """
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(documents)} documents to vector store")

        try:
            if self.vectorstore is None:
                # Première initialisation: création de la collection
                logger.info("Creating new vector store")
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
            else:
                # Ajout à une collection existante
                logger.info("Adding to existing vector store")
                self.vectorstore.add_documents(documents)

            # Sauvegarde sur disque
            self.vectorstore.persist()
            logger.info(f"Successfully added {len(documents)} documents")

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def load_existing(self) -> bool:
        """
        Charge une base vectorielle existante depuis le disque

        Returns:
            True si chargement réussi, False sinon

        Utile pour:
            - Reprendre une session précédente
            - Éviter de re-indexer les mêmes documents
            - Démarrage plus rapide de l'application
        """
        try:
            # Vérifie si le répertoire existe et contient des données
            if not os.path.exists(self.persist_directory):
                logger.info("No existing vector store found")
                return False

            # Tentative de chargement
            logger.info(f"Loading existing vector store from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )

            # Vérification que la collection contient des données
            collection = self.vectorstore._collection
            count = collection.count()

            if count == 0:
                logger.warning("Vector store exists but is empty")
                return False

            logger.info(f"Successfully loaded vector store with {count} embeddings")
            return True

        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        Recherche les documents les plus similaires à une question

        Args:
            query: Question de l'utilisateur
            k: Nombre de résultats à retourner (top-k)
            filter_dict: Filtres optionnels sur les métadonnées

        Returns:
            Liste des k documents les plus pertinents

        Comment ça marche:
            1. Question → Embedding (vecteur)
            2. Calcul de similarité avec tous les embeddings de la DB
            3. Retourne les k plus similaires

        Mesure de similarité:
            - Cosine similarity (angle entre vecteurs)
            - Plus l'angle est petit, plus les textes sont similaires
            - Score entre 0 (pas similaire) et 1 (identique)

        Pourquoi k=4 par défaut ?
            - Compromis entre contexte riche et bruit
            - 4 chunks ≈ 4000 caractères ≈ 1000 tokens
            - Rentre dans le contexte de la plupart des LLM
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Add documents first.")

        logger.info(f"Searching for: '{query}' (top-{k})")

        try:
            # La magie opère ici !
            # ChromaDB calcule les similarités et retourne les meilleurs matches
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )

            logger.info(f"Found {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple[Document, float]]:
        """
        Comme similarity_search mais retourne aussi les scores

        Returns:
            Liste de tuples (document, score)
            - score: distance/similarité (plus petit = plus similaire)

        Utile pour:
            - Debug: voir à quel point les résultats sont pertinents
            - Filtrage: ignorer les résultats avec score trop bas
            - UI: afficher un % de pertinence à l'utilisateur
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        logger.info(f"Searching with scores for: '{query}'")

        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )

            # Log des scores pour debug
            for i, (doc, score) in enumerate(results, 1):
                logger.debug(f"Result {i}: score={score:.4f}, source={doc.metadata.get('source', 'unknown')}")

            return results

        except Exception as e:
            logger.error(f"Error during search with scores: {str(e)}")
            raise

    def delete_collection(self) -> None:
        """
        Supprime complètement la collection

        Attention: Irréversible !

        Utile pour:
            - Reset complet
            - Changer de configuration d'embeddings
            - Nettoyer l'espace disque
        """
        logger.warning("Deleting vector store collection")

        try:
            if self.vectorstore is not None:
                self.vectorstore.delete_collection()
                self.vectorstore = None

            logger.info("Collection deleted successfully")

        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    def get_collection_stats(self) -> dict:
        """
        Retourne des statistiques sur la collection

        Returns:
            Dict avec: nombre de documents, taille, etc.

        Utile pour:
            - Dashboard
            - Monitoring
            - Debug
        """
        if self.vectorstore is None:
            return {"status": "not_initialized", "count": 0}

        try:
            collection = self.vectorstore._collection
            count = collection.count()

            return {
                "status": "active",
                "count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }

        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"status": "error", "error": str(e)}


# Exemple d'utilisation
if __name__ == "__main__":
    """
    Code de test/démo
    """
    # Création du vector store
    vs = VectorStore(use_ollama=False)  # Mettre True pour Ollama

    # Exemple de documents
    from langchain.schema import Document

    docs = [
        Document(page_content="Python est un langage de programmation", metadata={"source": "doc1"}),
        Document(page_content="JavaScript est utilisé pour le web", metadata={"source": "doc2"}),
    ]

    # Ajout des documents
    # vs.add_documents(docs)

    # Recherche
    # results = vs.similarity_search("langage de programmation", k=2)
    # for doc in results:
    #     print(doc.page_content)
