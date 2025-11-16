"""
RAG Engine - Cœur du système de Retrieval Augmented Generation
Ce module orchestre tout le pipeline RAG: retrieval + génération de réponse
"""

import logging
from typing import List, Optional, Dict, Any
import os

# LangChain imports
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.schema import Document

from src.vector_store import VectorStore
from src.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Moteur RAG principal - Le cerveau de DocuMind AI

    RAG = Retrieval Augmented Generation
    C'est quoi ?
        1. RETRIEVAL: Chercher les infos pertinentes dans la base vectorielle
        2. AUGMENTED: Enrichir le prompt avec ces infos
        3. GENERATION: Générer une réponse avec le LLM

    Pourquoi RAG ?
        - LLM seul: Connaissances limitées (date de coupure) + hallucinations
        - RAG: LLM + tes propres documents = réponses précises et sourçées
        - Exemple: ChatGPT ne connaît pas ta doc interne, RAG si !

    Pipeline complet:
        Question → Vector Search → Top-K docs → Prompt + Docs → LLM → Réponse
    """

    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
        use_ollama: bool = False
    ):
        """
        Initialise le moteur RAG

        Args:
            vector_store: Instance du VectorStore (déjà initialisée)
            model_name: Nom du modèle LLM
            temperature: Créativité du modèle (0=déterministe, 1=créatif)
            max_tokens: Longueur max de la réponse
            use_ollama: Utiliser Ollama (local) ou OpenAI

        Temperature expliquée:
            - 0.0: Réponses déterministes, factuelles (pour FAQ, doc technique)
            - 0.7: Équilibré (bon pour usage général)
            - 1.0+: Créatif, varié (pour brainstorming, création)

        Max tokens:
            - 500 tokens ≈ 2000 caractères ≈ 1 paragraphe détaillé
            - Plus = réponses longues mais + cher
        """
        self.vector_store = vector_store
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialisation du LLM
        if use_ollama:
            logger.info(f"Using Ollama model: {model_name}")
            # Ollama: LLM local, gratuit
            # Modèles populaires: llama2, mistral, codellama
            self.llm = Ollama(
                model=model_name,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=temperature
            )
        else:
            logger.info(f"Using OpenAI model: {model_name}")
            # ChatOpenAI: Wrapper LangChain pour GPT-3.5/GPT-4
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )

        # Template du prompt (super important !)
        # C'est lui qui guide le comportement du LLM
        self.prompt_template = self._create_prompt_template()

        # Chain RetrievalQA: Chaîne LangChain qui fait tout le RAG
        self.qa_chain = None
        self._initialize_qa_chain()

        logger.info("RAG Engine initialized successfully")

    def _create_prompt_template(self) -> PromptTemplate:
        """
        Crée le template de prompt pour le LLM

        Le prompt template est CRUCIAL pour la qualité des réponses !

        Structure du prompt:
            1. Instructions système (rôle, comportement)
            2. Contexte (documents récupérés)
            3. Question de l'utilisateur
            4. Instructions de réponse

        Bonnes pratiques:
            - Être explicite sur le rôle
            - Demander des sources
            - Gérer le cas "je ne sais pas"
            - Format de réponse structuré
        """

        template = """Tu es un assistant IA spécialisé dans l'analyse de documentation technique.
Ton rôle est de répondre aux questions en te basant UNIQUEMENT sur le contexte fourni ci-dessous.

RÈGLES IMPORTANTES:
1. Base tes réponses UNIQUEMENT sur le contexte fourni
2. Si l'information n'est pas dans le contexte, dis "Je ne trouve pas cette information dans la documentation fournie"
3. Cite toujours tes sources (nom du fichier, page si disponible)
4. Sois précis et concis
5. Si tu n'es pas sûr, dis-le clairement

CONTEXTE (Documents pertinents):
{context}

QUESTION DE L'UTILISATEUR:
{question}

RÉPONSE:
(Réponds de manière claire et structurée, avec les sources)
"""

        # PromptTemplate: Objet LangChain qui gère les variables {context} et {question}
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _initialize_qa_chain(self) -> None:
        """
        Initialise la chaîne RetrievalQA

        RetrievalQA = Chaîne LangChain qui:
            1. Prend une question
            2. Récupère les docs pertinents (via le retriever)
            3. Construit le prompt avec le template
            4. Envoie au LLM
            5. Retourne la réponse

        C'est la "magie" de LangChain: tout est orchestré automatiquement !
        """
        if self.vector_store.vectorstore is None:
            logger.warning("Vector store not initialized. QA chain will be created when needed.")
            return

        try:
            # Création du retriever depuis le vector store
            # retriever = composant qui fait la recherche vectorielle
            retriever = self.vector_store.vectorstore.as_retriever(
                search_kwargs={"k": 4}  # Top-4 documents les plus pertinents
            )

            # Création de la chaîne QA
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # "stuff" = met tous les docs dans le prompt
                retriever=retriever,
                return_source_documents=True,  # Retourne aussi les sources
                chain_type_kwargs={"prompt": self.prompt_template}
            )

            logger.info("QA chain initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing QA chain: {str(e)}")
            raise

    def query(self, question: str) -> Dict[str, Any]:
        """
        Pose une question au système RAG

        Args:
            question: Question de l'utilisateur

        Returns:
            Dict contenant:
                - answer: Réponse générée
                - sources: Documents sources utilisés
                - metadata: Infos additionnelles

        Pipeline détaillé:
            1. Question → Embedding
            2. Recherche vectorielle → Top-K docs
            3. Construction du prompt: Template + Docs + Question
            4. Appel LLM avec le prompt
            5. Parse de la réponse
            6. Retour structuré avec réponse + sources

        Exemple:
            >>> engine.query("Comment installer Python?")
            {
                "answer": "Pour installer Python, suivez ces étapes...",
                "sources": [Document(...), Document(...)],
                "metadata": {...}
            }
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Please load or add documents first.")

        logger.info(f"Processing query: {question}")

        try:
            # Appel de la chaîne QA
            # Tout le pipeline RAG se passe ici !
            result = self.qa_chain({"query": question})

            # Extraction de la réponse et des sources
            answer = result.get("result", "")
            source_documents = result.get("source_documents", [])

            logger.info(f"Query processed successfully. Answer length: {len(answer)} chars")

            # Construction de la réponse structurée
            response = {
                "answer": answer,
                "sources": self._format_sources(source_documents),
                "source_documents": source_documents,  # Documents complets (pour debug)
                "metadata": {
                    "num_sources": len(source_documents),
                    "model": self.llm.model_name if hasattr(self.llm, 'model_name') else "unknown",
                    "temperature": self.temperature
                }
            }

            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Formate les documents sources pour un affichage propre

        Args:
            documents: Liste de Documents LangChain

        Returns:
            Liste de dicts avec infos formatées

        Pourquoi formater ?
            - UI plus propre
            - Facile à afficher dans Streamlit
            - Sépare le contenu des métadonnées
        """
        formatted_sources = []

        for i, doc in enumerate(documents, 1):
            source_info = {
                "id": i,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "full_content": doc.page_content,
                "metadata": doc.metadata,
                "source_file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            }
            formatted_sources.append(source_info)

        return formatted_sources

    def add_documents_to_knowledge_base(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Ajoute des documents à la base de connaissances

        Args:
            file_paths: Liste de chemins vers les fichiers à ajouter

        Returns:
            Dict avec stats sur l'indexation

        Process:
            1. DocumentProcessor charge et découpe les fichiers
            2. VectorStore crée les embeddings et stocke
            3. Re-initialise la QA chain avec la nouvelle DB

        C'est la méthode pour "nourrir" ton RAG avec de nouvelles docs !
        """
        logger.info(f"Adding {len(file_paths)} file(s) to knowledge base")

        processor = DocumentProcessor()
        all_chunks = []
        stats = {
            "total_files": len(file_paths),
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "errors": []
        }

        # Traitement de chaque fichier
        for file_path in file_paths:
            try:
                chunks = processor.process_file(file_path)
                all_chunks.extend(chunks)
                stats["processed_files"] += 1

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                stats["failed_files"] += 1
                stats["errors"].append({"file": file_path, "error": str(e)})

        # Ajout à la base vectorielle
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            stats["total_chunks"] = len(all_chunks)

            # Re-initialisation de la QA chain
            self._initialize_qa_chain()

        logger.info(f"Knowledge base updated: {stats}")
        return stats

    def get_relevant_documents(self, question: str, k: int = 4) -> List[Document]:
        """
        Récupère les documents pertinents sans générer de réponse

        Args:
            question: Question/requête
            k: Nombre de documents à retourner

        Returns:
            Liste des documents les plus pertinents

        Utile pour:
            - Debug: voir ce que le RAG récupère
            - UI: afficher les sources avant la réponse
            - Analyse: évaluer la qualité du retrieval
        """
        logger.info(f"Retrieving relevant documents for: {question}")

        return self.vector_store.similarity_search(question, k=k)


# Exemple d'utilisation
if __name__ == "__main__":
    """
    Code de test/démo
    """
    # Setup
    from dotenv import load_dotenv
    load_dotenv()

    # Création des composants
    vector_store = VectorStore()
    # vector_store.load_existing()  # Charge DB existante

    # Création du RAG engine
    # rag = RAGEngine(vector_store, use_ollama=False)

    # Ajout de documents
    # rag.add_documents_to_knowledge_base(["doc1.pdf", "doc2.md"])

    # Question
    # response = rag.query("Comment fonctionne Python?")
    # print(response["answer"])
    # print(f"\nSources: {len(response['sources'])}")
