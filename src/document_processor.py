"""
Document Processor - Traitement et découpage des documents
Ce module gère le chargement et la préparation des documents pour le RAG
"""

import logging
from pathlib import Path
from typing import List, Optional
import os

# LangChain imports pour le traitement de documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)
from langchain.schema import Document

from src.utils import is_supported_file, get_file_extension

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Classe qui gère le traitement des documents

    Responsabilités:
        1. Charger différents types de fichiers (PDF, TXT, MD, DOCX)
        2. Découper les documents en chunks (morceaux) optimaux
        3. Préserver les métadonnées (nom de fichier, numéro de page, etc.)

    Pourquoi découper en chunks ?
        - Les LLM ont une limite de tokens (context window)
        - Les petits chunks permettent une recherche plus précise
        - Meilleur retrieval = meilleures réponses
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialise le processeur de documents

        Args:
            chunk_size: Taille maximale d'un chunk en caractères
                - 1000 caractères ≈ 250 tokens
                - Assez grand pour du contexte, assez petit pour être précis

            chunk_overlap: Chevauchement entre chunks
                - 200 caractères de overlap = continuité entre chunks
                - Évite de couper des informations importantes
                - Exemple: si une phrase est à cheval sur 2 chunks,
                  l'overlap garantit qu'elle sera complète dans au moins un chunk

        Pourquoi ces valeurs par défaut ?
            - chunk_size=1000: Sweet spot entre contexte et précision
            - chunk_overlap=200: 20% d'overlap est une bonne pratique
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # RecursiveCharacterTextSplitter : Le meilleur text splitter de LangChain
        # "Recursive" = essaie de découper intelligemment (paragraphes, phrases, mots)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # Fonction pour mesurer la longueur
            separators=["\n\n", "\n", ". ", " ", ""]  # Ordre de priorité pour couper
        )

        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def load_document(self, file_path: str) -> List[Document]:
        """
        Charge un document et retourne une liste de Documents LangChain

        Args:
            file_path: Chemin vers le fichier à charger

        Returns:
            Liste de Documents (objets LangChain contenant texte + métadonnées)

        Raises:
            ValueError: Si le format de fichier n'est pas supporté
            FileNotFoundError: Si le fichier n'existe pas

        Comment ça marche:
            1. Vérifie que le fichier existe
            2. Détecte l'extension (.pdf, .txt, etc.)
            3. Utilise le bon loader selon l'extension
            4. Retourne le contenu chargé
        """
        # Vérifications de base
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not is_supported_file(file_path):
            raise ValueError(f"Unsupported file type: {file_path}")

        # Sélection du bon loader selon l'extension
        extension = get_file_extension(file_path)
        logger.info(f"Loading document: {file_path} (type: {extension})")

        try:
            # Chaque loader est spécialisé pour un format
            if extension == '.pdf':
                # PyPDFLoader: extrait le texte des PDFs + numéro de page
                loader = PyPDFLoader(file_path)
            elif extension == '.txt':
                # TextLoader: simple lecture de fichier texte
                loader = TextLoader(file_path, encoding='utf-8')
            elif extension == '.md':
                # UnstructuredMarkdownLoader: préserve la structure Markdown
                loader = UnstructuredMarkdownLoader(file_path)
            elif extension == '.docx':
                # Docx2txtLoader: extrait le texte des fichiers Word
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported extension: {extension}")

            # Chargement du document
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} document(s) from {file_path}")

            # Ajout de métadonnées supplémentaires
            for doc in documents:
                doc.metadata['source'] = os.path.basename(file_path)
                doc.metadata['file_path'] = file_path

            return documents

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Découpe une liste de documents en chunks optimaux

        Args:
            documents: Liste de documents à découper

        Returns:
            Liste de chunks (sous-documents)

        Exemple:
            Input: 1 document de 10,000 caractères
            Output: ~10 chunks de ~1000 caractères chacun (avec overlap)

        Pourquoi c'est important:
            - Chunks = unités de base pour la recherche vectorielle
            - Taille optimale = meilleur retrieval
            - Overlap = pas de perte d'information entre chunks
        """
        logger.info(f"Splitting {len(documents)} document(s) into chunks")

        # Le text_splitter fait tout le travail intelligent
        chunks = self.text_splitter.split_documents(documents)

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} document(s)")

        # Ajout d'un ID unique à chaque chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i

        return chunks

    def process_file(self, file_path: str) -> List[Document]:
        """
        Méthode tout-en-un : charge et découpe un fichier

        Args:
            file_path: Chemin vers le fichier

        Returns:
            Liste de chunks prêts pour l'indexation

        C'est la méthode principale que tu utiliseras dans ton code !

        Pipeline complet:
            Fichier → Load → Documents → Split → Chunks
        """
        logger.info(f"Processing file: {file_path}")

        # Étape 1: Chargement
        documents = self.load_document(file_path)

        # Étape 2: Découpage
        chunks = self.split_documents(documents)

        logger.info(f"File processed: {file_path} -> {len(chunks)} chunks")

        return chunks

    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Traite tous les fichiers supportés dans un répertoire

        Args:
            directory_path: Chemin vers le répertoire

        Returns:
            Liste de tous les chunks de tous les fichiers

        Utile pour:
            - Indexer toute une documentation d'un coup
            - Batch processing de plusieurs fichiers
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Not a directory: {directory_path}")

        all_chunks = []
        processed_files = 0

        # Parcours de tous les fichiers du répertoire
        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)

                # Skip les fichiers non supportés
                if not is_supported_file(file_path):
                    logger.debug(f"Skipping unsupported file: {filename}")
                    continue

                try:
                    # Traitement du fichier
                    chunks = self.process_file(file_path)
                    all_chunks.extend(chunks)
                    processed_files += 1

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {str(e)}")
                    continue

        logger.info(f"Processed {processed_files} files from {directory_path}")
        logger.info(f"Total chunks created: {len(all_chunks)}")

        return all_chunks


# Exemple d'utilisation (pour comprendre)
if __name__ == "__main__":
    """
    Ce code s'exécute seulement si tu lances ce fichier directement
    Utile pour tester le module isolément
    """
    # Création du processeur
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

    # Test avec un fichier (exemple)
    # chunks = processor.process_file("example.pdf")
    # print(f"Created {len(chunks)} chunks")
    # print(f"First chunk: {chunks[0].page_content[:200]}...")
