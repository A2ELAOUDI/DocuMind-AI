# Guide Python - Documentation d'Exemple

## Introduction à Python

Python est un langage de programmation interprété, de haut niveau et à usage général. Créé par Guido van Rossum et publié pour la première fois en 1991, Python met l'accent sur la lisibilité du code.

### Caractéristiques Principales

- **Syntaxe claire et lisible**: Python utilise l'indentation pour délimiter les blocs de code
- **Multi-paradigme**: Supporte la programmation orientée objet, impérative et fonctionnelle
- **Typage dynamique**: Pas besoin de déclarer les types de variables
- **Grande bibliothèque standard**: Batteries incluses
- **Communauté active**: Énorme écosystème de packages tiers

## Installation de Python

### Windows

1. Téléchargez l'installateur depuis [python.org](https://python.org)
2. Exécutez le fichier `.exe`
3. **Important**: Cochez la case "Add Python to PATH"
4. Cliquez sur "Install Now"
5. Vérifiez l'installation avec `python --version`

### macOS

```bash
# Avec Homebrew
brew install python3

# Vérification
python3 --version
```

### Linux (Ubuntu/Debian)

```bash
# Installation
sudo apt update
sudo apt install python3 python3-pip

# Vérification
python3 --version
```

## Variables et Types de Données

### Déclaration de Variables

En Python, pas besoin de déclarer le type explicitement:

```python
# Nombres
age = 25
price = 19.99

# Chaînes de caractères
name = "Alice"
message = 'Hello World'

# Booléens
is_active = True
is_verified = False

# Listes
fruits = ["pomme", "banane", "orange"]

# Dictionnaires
person = {
    "name": "Bob",
    "age": 30,
    "city": "Paris"
}
```

### Types Principaux

1. **int**: Nombres entiers
2. **float**: Nombres décimaux
3. **str**: Chaînes de caractères
4. **bool**: Booléens (True/False)
5. **list**: Listes ordonnées modifiables
6. **tuple**: Listes ordonnées immuables
7. **dict**: Dictionnaires (paires clé-valeur)
8. **set**: Ensembles (valeurs uniques)

## Structures de Contrôle

### Conditions (if/elif/else)

```python
age = 18

if age < 18:
    print("Mineur")
elif age == 18:
    print("Tout juste majeur")
else:
    print("Majeur")
```

### Boucles

#### Boucle for

```python
# Itération sur une liste
fruits = ["pomme", "banane", "orange"]
for fruit in fruits:
    print(fruit)

# Utilisation de range
for i in range(5):
    print(i)  # Affiche 0, 1, 2, 3, 4
```

#### Boucle while

```python
count = 0
while count < 5:
    print(count)
    count += 1
```

## Fonctions

### Définition d'une Fonction

```python
def saluer(nom):
    """Fonction qui salue une personne"""
    return f"Bonjour, {nom}!"

# Utilisation
message = saluer("Alice")
print(message)  # Affiche: Bonjour, Alice!
```

### Paramètres par Défaut

```python
def calculer_prix(prix, taxe=0.20):
    """Calcule le prix TTC avec taxe par défaut de 20%"""
    return prix * (1 + taxe)

# Avec taxe par défaut
prix_ttc = calculer_prix(100)  # 120.0

# Avec taxe personnalisée
prix_ttc = calculer_prix(100, 0.10)  # 110.0
```

## Programmation Orientée Objet

### Définition d'une Classe

```python
class Personne:
    """Représente une personne"""

    def __init__(self, nom, age):
        """Constructeur"""
        self.nom = nom
        self.age = age

    def se_presenter(self):
        """Méthode d'instance"""
        return f"Je m'appelle {self.nom} et j'ai {self.age} ans"

# Création d'une instance
alice = Personne("Alice", 25)
print(alice.se_presenter())
```

### Héritage

```python
class Etudiant(Personne):
    """Classe dérivée de Personne"""

    def __init__(self, nom, age, universite):
        super().__init__(nom, age)
        self.universite = universite

    def etudier(self):
        return f"{self.nom} étudie à {self.universite}"
```

## Gestion des Fichiers

### Lecture d'un Fichier

```python
# Méthode recommandée (with statement)
with open("fichier.txt", "r", encoding="utf-8") as f:
    contenu = f.read()
    print(contenu)

# Le fichier est automatiquement fermé
```

### Écriture dans un Fichier

```python
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello World\\n")
    f.write("Deuxième ligne\\n")
```

## Gestion des Erreurs

### Try/Except

```python
try:
    resultat = 10 / 0
except ZeroDivisionError:
    print("Erreur: Division par zéro!")
except Exception as e:
    print(f"Erreur inattendue: {e}")
finally:
    print("Ce bloc s'exécute toujours")
```

## Modules et Packages

### Import de Modules

```python
# Import complet
import math
print(math.pi)

# Import spécifique
from math import sqrt, pi
print(sqrt(16))

# Import avec alias
import numpy as np
array = np.array([1, 2, 3])
```

### Installer des Packages

```bash
# Avec pip
pip install requests
pip install pandas numpy

# Depuis requirements.txt
pip install -r requirements.txt
```

## Compréhensions de Listes

### List Comprehension

```python
# Méthode classique
nombres_carres = []
for i in range(10):
    nombres_carres.append(i ** 2)

# Avec list comprehension (plus pythonique)
nombres_carres = [i ** 2 for i in range(10)]

# Avec condition
nombres_pairs = [i for i in range(20) if i % 2 == 0]
```

### Dict Comprehension

```python
# Créer un dictionnaire
carres = {i: i**2 for i in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

## Bonnes Pratiques

### Style de Code (PEP 8)

1. **Indentation**: 4 espaces (pas de tabs)
2. **Noms de variables**: `snake_case` (minuscules avec underscores)
3. **Noms de classes**: `PascalCase` (première lettre en majuscule)
4. **Constantes**: `MAJUSCULES_AVEC_UNDERSCORES`
5. **Longueur de ligne**: Maximum 79 caractères

### Documentation

```python
def ma_fonction(param1, param2):
    """
    Résumé de la fonction.

    Args:
        param1 (int): Description du paramètre 1
        param2 (str): Description du paramètre 2

    Returns:
        bool: Description du retour

    Raises:
        ValueError: Si param1 est négatif
    """
    pass
```

## Ressources Utiles

- **Documentation officielle**: https://docs.python.org/
- **PyPI** (Python Package Index): https://pypi.org/
- **Real Python**: https://realpython.com/
- **Python.org**: https://python.org/

## Conclusion

Python est un excellent langage pour débuter en programmation grâce à sa syntaxe claire et sa grande communauté. Que ce soit pour le développement web, l'analyse de données, l'intelligence artificielle ou l'automatisation, Python a les outils nécessaires.

**Prochaines étapes recommandées:**
1. Pratiquer avec des exercices (Codewars, LeetCode)
2. Créer des petits projets personnels
3. Lire du code open-source
4. Contribuer à la communauté

Bon apprentissage ! 🐍
