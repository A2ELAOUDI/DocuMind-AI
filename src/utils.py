"""
Utility functions for DocuMind AI
Ce fichier contient des fonctions utilitaires réutilisables dans tout le projet
"""

import os
from typing import Optional
from pathlib import Path
import logging

# Configuration du logging pour afficher les messages de débogage
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_directory_exists(directory_path: str) -> None:
    """
    Crée un répertoire s'il n'existe pas déjà

    Args:
        directory_path: Chemin du répertoire à créer

    Pourquoi c'est utile:
        - Évite les erreurs quand on essaie d'écrire dans un dossier qui n'existe pas
        - Utilisé pour créer le dossier de la base vectorielle
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory ensured: {directory_path}")


def get_file_extension(file_path: str) -> str:
    """
    Extrait l'extension d'un fichier

    Args:
        file_path: Chemin du fichier

    Returns:
        Extension du fichier (ex: '.pdf', '.txt', '.md')

    Exemple:
        get_file_extension("document.pdf") -> ".pdf"
    """
    return Path(file_path).suffix.lower()


def is_supported_file(file_path: str) -> bool:
    """
    Vérifie si le fichier est supporté par notre système

    Args:
        file_path: Chemin du fichier à vérifier

    Returns:
        True si le fichier est supporté, False sinon

    Formats supportés:
        - PDF (.pdf)
        - Word (.docx)
        - Texte (.txt)
        - Markdown (.md)
    """
    supported_extensions = {'.pdf', '.txt', '.md', '.docx'}
    extension = get_file_extension(file_path)
    return extension in supported_extensions


def count_tokens_approximate(text: str) -> int:
    """
    Estime le nombre de tokens dans un texte

    Args:
        text: Texte à analyser

    Returns:
        Nombre approximatif de tokens

    Note:
        Utilise une approximation simple: 1 token ≈ 4 caractères
        Pour une mesure précise, il faudrait utiliser tiktoken d'OpenAI

    Pourquoi c'est important:
        - Les API LLM facturent au token
        - Il faut respecter les limites de contexte (ex: 4096 tokens pour GPT-3.5)
    """
    return len(text) // 4


def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Tronque un texte à une longueur maximale

    Args:
        text: Texte à tronquer
        max_length: Longueur maximale en caractères

    Returns:
        Texte tronqué avec "..." si nécessaire
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_sources(sources: list) -> str:
    """
    Formate les sources pour un affichage propre

    Args:
        sources: Liste de documents sources

    Returns:
        String formaté avec les sources

    Exemple de sortie:
        Source 1: document.pdf (page 5)
        Source 2: guide.md (ligne 42)
    """
    if not sources:
        return "Aucune source trouvée"

    formatted = []
    for i, source in enumerate(sources, 1):
        metadata = source.metadata
        filename = metadata.get('source', 'Unknown')
        page = metadata.get('page', 'N/A')
        formatted.append(f"Source {i}: {filename} (page {page})")

    return "\n".join(formatted)


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Valide qu'une clé API est présente et non vide

    Args:
        api_key: Clé API à valider

    Returns:
        True si la clé est valide, False sinon

    Pourquoi c'est important:
        - Évite des erreurs cryptiques si la clé API est manquante
        - Permet de donner un message d'erreur clair à l'utilisateur
    """
    if not api_key or api_key.strip() == "":
        logger.error("API key is missing or empty")
        return False

    if api_key == "your_openai_api_key_here":
        logger.error("API key has not been configured (still default value)")
        return False

    return True


def sanitize_filename(filename: str) -> str:
    """
    Nettoie un nom de fichier pour le rendre safe

    Args:
        filename: Nom de fichier à nettoyer

    Returns:
        Nom de fichier nettoyé (sans caractères dangereux)

    Pourquoi:
        - Évite les problèmes avec des caractères spéciaux
        - Prévient les attaques par injection de path (../../../etc/passwd)
    """
    # Supprime les caractères dangereux
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    return filename.strip()
