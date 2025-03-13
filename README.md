# 🌐 Embedding API Gateway 🚀

## 🔥 Passerelle universelle pour tous vos services d'embeddings et recherche sémantique vectorielle 🔥

[![GitHub stars](https://img.shields.io/github/stars/simonpierreboucher0/embedding-api?style=social)](https://github.com/simonpierreboucher0/embedding-api/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.105.0+-green.svg)](https://fastapi.tiangolo.com/)

---

## ✨ Caractéristiques principales

🔄 **Interface unifiée** - Une API pour tous les fournisseurs d'embeddings  
🧩 **Triple moteur de recherche vectorielle** - FAISS, ChromaDB ou LanceDB au choix  
🔍 **Recherche vectorielle avancée** - Recherche sémantique optimisée pour différents cas d'usage  
📊 **Stockage flexible** - Formats optimisés pour rapidité, lisibilité ou évolutivité  
⚡ **Hautes performances** - Optimisations pour grands volumes de données  
🧠 **Gestion d'index** - Création, mise à jour et suppression simplifiées  
🔒 **Sécurité intégrée** - Gestion sécurisée des clés API  
🌈 **Comparaison directe** - Calcul instantané de similarité entre textes  

---

## 🤖 Fournisseurs et modèles pris en charge

### 🟢 OpenAI
Modèles d'embeddings leaders du marché.

| Modèle | Dimension | Cas d'utilisation |
|--------|-----------|-------------------|
| 🌟 **text-embedding-ada-002** | 1536 | Recherche sémantique, classification, clustering |
| 💎 **text-embedding-3-small** | 1536 | Applications avec contraintes de coût, performance équilibrée |
| 🏆 **text-embedding-3-large** | 3072 | Recherche haute précision, applications premium |

### 🟣 Cohere
Modèles d'embeddings spécialisés et multilingues.

| Modèle | Cas d'utilisation |
|--------|-------------------|
| 🌠 **embed-english-v3.0** | Recherche sémantique optimisée pour l'anglais |
| 🌍 **embed-multilingual-v3.0** | Applications multilingues, recherche cross-language |
| ⚡ **embed-english-light-v3.0** | Version légère et rapide pour l'anglais |
| 🚀 **embed-multilingual-light-v3.0** | Version légère multilingue, faible latence |

### 🔵 Mistral AI
Modèle d'embedding puissant conçu en France.

| Modèle | Cas d'utilisation |
|--------|-------------------|
| 💫 **mistral-embed** | Recherche sémantique, RAG, classification de documents |

### 🟡 DeepInfra
Large sélection de modèles d'embeddings open-source optimisés.

| Modèle | Cas d'utilisation |
|--------|-------------------|
| 🔍 **BAAI/bge-base-en-v1.5** | Modèle équilibré pour l'anglais |
| 🔎 **BAAI/bge-small-en-v1.5** | Version compacte, faible empreinte mémoire |
| 🔬 **BAAI/bge-large-en-v1.5** | Précision élevée pour applications critiques |
| 🌟 **thenlper/gte-large** | Modèle général avec large couverture sémantique |
| ⭐ **thenlper/gte-base** | Version équilibrée pour usage général |
| 📊 **intfloat/e5-large-v2** | Optimisé pour RAG et recherche avancée |
| 🧮 **sentence-transformers/all-mpnet-base-v2** | Classique pour transformation de phrases |

### 🟠 Voyage AI
Modèles d'embeddings spécialisés avec capacités avancées.

| Modèle | Cas d'utilisation |
|--------|-------------------|
| 🌊 **voyage-large-2** | Recherche sémantique de pointe, applications premium |
| 🧠 **voyage-large-2-instruct** | Recherche sémantique guidée par instructions |
| 💻 **voyage-code-2** | Spécialisé pour code source et documentation technique |

---

## 🔄 Bases de données vectorielles supportées

### 🌟 FAISS (Facebook AI Similarity Search)
Bibliothèque de recherche vectorielle ultra-rapide.

| Points forts | Cas d'utilisation |
|--------------|-------------------|
| ⚡ **Ultra performant** | Grands volumes de données, recherche en temps réel |
| 📏 **Recherche par distance euclidienne** | Précision élevée pour la recherche de similitude |
| 🚀 **Optimisé pour le calcul distribué** | Applications à l'échelle de production |

### 💠 Chroma
Base de données vectorielle conçue pour les applications d'IA.

| Points forts | Cas d'utilisation |
|--------------|-------------------|
| 🧩 **API intuitive** | Développement rapide d'applications RAG |
| 🌐 **Flexibilité des métadonnées** | Filtrage complexe, recherche hybride |
| 📊 **Collections et espaces de noms** | Organisation efficace des données |

### 📊 LanceDB
Base de données vectorielle orientée document avec persistance.

| Points forts | Cas d'utilisation |
|--------------|-------------------|
| 💾 **Persistance intégrée** | Données conservées entre les redémarrages |
| 📁 **Format Apache Arrow** | Performance et interopérabilité élevées |
| 🔄 **Mises à jour performantes** | Index évolutifs et modifiables |

### 📐 Similarité Cosinus
Méthode simple de comparaison vectorielle (incluse pour compatibilité).

| Points forts | Cas d'utilisation |
|--------------|-------------------|
| 🔍 **Transparence** | Stockage JSON lisible et facilement inspectable |
| 🧪 **Simplicité** | Prototypage, petits projets, tests |
| 🧮 **Précision de base** | Comparaisons directes sans optimisation spéciale |

---

## 🛠️ Installation facile en 3 étapes

### 1️⃣ Cloner le dépôt
```bash
git clone https://github.com/simonpierreboucher0/embedding-api.git
cd embedding-api
```

### 2️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3️⃣ Configurer vos clés API
Créez un fichier `.env` avec vos clés:
```ini
OPENAI_API_KEY=sk-xxxx
COHERE_API_KEY=xxxx
MISTRAL_API_KEY=xxxx
DEEPINFRA_API_KEY=xxxx
VOYAGE_API_KEY=xxxx
```

> 💡 **Astuce**: Vous n'avez besoin de fournir que les clés pour les fournisseurs que vous allez utiliser!

---

## 🚀 Guide d'utilisation rapide

### ▶️ Démarrer le serveur
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 📋 Obtenir la liste des modèles et bases de données vectorielles
```bash
curl -X GET http://localhost:8000/models
curl -X GET http://localhost:8000/vector_dbs
```

### 🧠 Générer des embeddings
```bash
curl -X POST http://localhost:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "text-embedding-3-small",
    "texts": "Artificial intelligence is transforming the world."
  }'
```

### 🗄️ Créer un index avec FAISS
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "text-embedding-3-small",
    "index_name": "ai_concepts",
    "texts": [
      "Artificial intelligence is the simulation of human intelligence processes by machines.",
      "Machine learning is a subset of AI that enables systems to learn from data.",
      "Deep learning is based on neural networks with many layers.",
      "Natural language processing allows machines to understand human language.",
      "Computer vision enables machines to interpret and make decisions based on visual input."
    ],
    "db_type": "faiss"
  }'
```

### 🌟 Créer un index avec ChromaDB
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "text-embedding-3-small",
    "index_name": "ai_concepts_chroma",
    "texts": [
      "Artificial intelligence is the simulation of human intelligence processes by machines.",
      "Machine learning is a subset of AI that enables systems to learn from data.",
      "Deep learning is based on neural networks with many layers.",
      "Natural language processing allows machines to understand human language.",
      "Computer vision enables machines to interpret and make decisions based on visual input."
    ],
    "db_type": "chroma"
  }'
```

### 📊 Créer un index avec LanceDB
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "text-embedding-3-small",
    "index_name": "ai_concepts_lance",
    "texts": [
      "Artificial intelligence is the simulation of human intelligence processes by machines.",
      "Machine learning is a subset of AI that enables systems to learn from data.",
      "Deep learning is based on neural networks with many layers.",
      "Natural language processing allows machines to understand human language.",
      "Computer vision enables machines to interpret and make decisions based on visual input."
    ],
    "db_type": "lancedb"
  }'
```

### 🔍 Recherche sémantique
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "text-embedding-3-small",
    "index_name": "ai_concepts",
    "query": "How do computers understand text?",
    "top_k": 3,
    "db_type": "faiss"
  }'
```

### 🔄 Mettre à jour un index existant
```bash
curl -X PUT http://localhost:8000/index/ai_concepts \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "text-embedding-3-small",
    "texts": [
      "Reinforcement learning is a training method based on rewarding desired behaviors.",
      "Transformer models have revolutionized natural language processing tasks."
    ],
    "db_type": "faiss"
  }'
```

### 🧮 Comparer deux textes directement
```bash
curl -X POST "http://localhost:8000/compare" \
  -d "provider=openai&model=text-embedding-3-small&texts=Artificial%20intelligence%20is%20transforming%20industries.&texts=AI%20is%20changing%20how%20businesses%20operate."
```

---

## 📊 Structure complète des requêtes et réponses

### 📝 Génération d'embeddings

**Requête:**
```json
{
  "provider": "openai",              // 🌐 Fournisseur (obligatoire)
  "model": "text-embedding-3-small", // 🤖 Modèle spécifique (obligatoire)
  "texts": ["Premier texte", "Deuxième texte"], // 📄 Textes (string ou array)
  "encoding_format": "float",        // 📊 Format d'encodage (pour certains providers)
  "input_type": "classification"     // 🏷️ Type d'entrée (pour Cohere)
}
```

**Réponse:**
```json
{
  "embeddings": [                    // 🧠 Liste des vecteurs d'embedding
    [0.0023, -0.0118, 0.0094, ...],  // 📊 Premier vecteur
    [0.0089, -0.0342, 0.0211, ...]   // 📊 Deuxième vecteur
  ],
  "model": "text-embedding-3-small", // 🤖 Modèle utilisé
  "provider": "openai",              // 🌐 Fournisseur utilisé
  "dimension": 1536,                 // 📏 Dimension des vecteurs
  "total_tokens": 14                 // 🔢 Nombre de tokens (si disponible)
}
```

### 🗄️ Création d'index

**Requête:**
```json
{
  "provider": "openai",              // 🌐 Fournisseur (obligatoire)
  "model": "text-embedding-3-small", // 🤖 Modèle spécifique (obligatoire)
  "index_name": "mon_index",         // 📚 Nom de l'index (obligatoire)
  "texts": [                         // 📄 Textes à indexer (obligatoire)
    "Premier document à indexer",
    "Deuxième document à indexer",
    "Troisième document à indexer"
  ],
  "db_type": "faiss",                // 💾 Type de base de données: "faiss", "chroma", "lancedb" ou "cosine"
  "encoding_format": "float",        // 📊 Format d'encodage (optionnel)
  "input_type": "classification"     // 🏷️ Type d'entrée (pour Cohere, optionnel)
}
```

**Réponse:**
```json
{
  "provider": "openai",              // 🌐 Fournisseur utilisé
  "model": "text-embedding-3-small", // 🤖 Modèle utilisé
  "index_name": "mon_index",         // 📚 Nom de l'index créé
  "total_chunks": 3,                 // 🧩 Nombre de documents indexés
  "created_at": "2023-08-15T14:23:45.123456", // ⏰ Date de création
  "dimension": 1536,                 // 📏 Dimension des vecteurs
  "db_type": "faiss"                 // 💾 Type de base de données utilisée
}
```

### 🔍 Recherche sémantique

**Requête:**
```json
{
  "provider": "openai",              // 🌐 Fournisseur (obligatoire)
  "model": "text-embedding-3-small", // 🤖 Modèle spécifique (obligatoire)
  "index_name": "mon_index",         // 📚 Nom de l'index (obligatoire)
  "query": "Ma requête de recherche", // 🔎 Texte de la requête (obligatoire)
  "top_k": 5,                        // 🔝 Nombre de résultats souhaités
  "db_type": "faiss"                 // 💾 Type de base de données: "faiss", "chroma", "lancedb" ou "cosine"
}
```

**Réponse:**
```json
{
  "index_name": "mon_index",         // 📚 Nom de l'index utilisé
  "provider": "openai",              // 🌐 Fournisseur utilisé
  "model": "text-embedding-3-small", // 🤖 Modèle utilisé
  "query": "Ma requête de recherche", // 🔎 Texte de la requête
  "db_type": "faiss",                // 💾 Type de base de données utilisée
  "results": [                       // 📋 Résultats de recherche
    {
      "chunk_id": 2,                 // 🆔 ID du document
      "text": "Deuxième document à indexer", // 📄 Texte du document
      "distance": 0.125,             // 📏 Distance (varie selon la base de données)
      "score": 0.89,                 // 📊 Score de similarité
      "rank": 1                      // 🏅 Rang dans les résultats
    },
    // ... autres résultats
  ]
}
```

---

## 🧪 Comparaison des bases de données vectorielles

### 🚀 FAISS (Facebook AI Similarity Search)

**Avantages:**
- ⚡ **Ultra rapide** pour les grands ensembles de données
- 🔍 **Recherche approximative** optimisée pour hautes dimensions
- 📈 **Passage à l'échelle** efficace avec des millions de vecteurs
- 🧠 **Optimisations mémoire** pour les grandes collections

**Utilisez FAISS quand:**
- 🗄️ Vous avez de grandes collections de documents (>10K textes)
- ⏱️ La vitesse de recherche est critique
- 💽 Vous avez des contraintes de mémoire pour de grands index

### 💠 ChromaDB

**Avantages:**
- 🔄 **Intégration native** avec des systèmes RAG
- 🧠 **Conçu pour l'IA** et optimisé pour les embeddings
- 🔍 **Recherche hybride** combinant vecteurs et métadonnées
- 📦 **Collections organisées** pour structurer vos données

**Utilisez ChromaDB quand:**
- 🤖 Vous développez des applications RAG complexes
- 📝 Vous avez besoin de filtres et métadonnées élaborés
- 📚 Vous gérez plusieurs collections liées
- 🔄 Vous voulez une solution adaptée aux workflows d'IA

### 📊 LanceDB

**Avantages:**
- 💽 **Persistance native** des données entre redémarrages
- 📁 **Format Apache Arrow** pour performance optimale
- 🔄 **Requêtes complexes** sur données vectorielles
- 🧩 **Orienté document** pour données structurées

**Utilisez LanceDB quand:**
- 💾 La persistance des données est essentielle
- 📚 Vous manipulez des documents avec structure complexe
- 🔄 Vous avez besoin de mises à jour fréquentes de l'index
- 📈 Vous cherchez un compromis entre performance et fonctionnalités

### 📐 Similarité Cosinus (avec stockage JSON)

**Avantages:**
- 📋 **Simple et transparent** - stockage en fichiers JSON lisibles
- 🎯 **Précision élevée** pour les petites collections
- 🔧 **Facile à déboguer** et à inspecter manuellement
- 🔄 **Portable** entre différents environnements

**Utilisez Cosinus quand:**
- 📝 Vous avez de petites collections (< 10K textes)
- 👁️ La lisibilité/transparence des données est importante
- 🧪 Vous prototypez ou testez votre application

---

## 💡 Cas d'utilisation avancés

### 🔄 Chaîne de traitement RAG (Retrieval Augmented Generation)

```python
import requests
import json

# 1. Indexer des documents techniques
documents = [
    "FastAPI est un framework web moderne pour Python.",
    "FAISS est une bibliothèque pour la recherche de similarité efficace.",
    "Les embeddings vectoriels représentent le texte dans un espace multidimensionnel.",
    "La recherche sémantique permet de trouver des informations par leur signification.",
    "La distance cosinus est une mesure courante de similarité entre vecteurs."
]

# Créer l'index avec ChromaDB pour une utilisation optimale dans les flux RAG
requests.post(
    "http://localhost:8000/index",
    json={
        "provider": "openai",
        "model": "text-embedding-3-small",
        "index_name": "docs_techniques",
        "texts": documents,
        "db_type": "chroma"
    }
)

# 2. Quand l'utilisateur pose une question
user_query = "Comment fonctionne la recherche vectorielle?"

# 3. Rechercher les documents pertinents
search_response = requests.post(
    "http://localhost:8000/search",
    json={
        "provider": "openai",
        "model": "text-embedding-3-small",
        "index_name": "docs_techniques",
        "query": user_query,
        "top_k": 2,
        "db_type": "chroma"
    }
)

relevant_docs = [result["text"] for result in search_response.json()["results"]]

# 4. Construire un prompt enrichi pour un LLM
context = "\n".join(relevant_docs)
prompt = f"""
Contexte:
{context}

Question: {user_query}

Réponse basée sur le contexte donné:
"""

# 5. Envoyer à un LLM (via un service externe comme OpenAI)
# Code pour appeler un LLM avec le prompt enrichi
print("Prompt enrichi avec contexte:", prompt)
```

### 🧪 Test A/B de différentes bases de données vectorielles

```python
import requests
import time
import numpy as np
from tabulate import tabulate

# Bases de données vectorielles à comparer
db_types = ["faiss", "chroma", "lancedb", "cosine"]

# Documents de test
documents = [
    "Artificial intelligence is the simulation of human intelligence in machines.",
    "Machine learning algorithms improve automatically through experience.",
    "Neural networks are computing systems inspired by biological neural networks.",
    "Deep learning is part of a broader family of machine learning methods.",
    "Natural language processing helps computers understand human language.",
    "Computer vision enables machines to interpret visual information.",
    "Reinforcement learning is learning what to do to maximize a reward signal.",
    "Supervised learning uses labeled training data to learn the mapping function.",
    "Unsupervised learning finds patterns in data without pre-existing labels.",
    "Transfer learning reuses a pre-trained model on a new problem."
]

# Création des index
for db_type in db_types:
    requests.post(
        "http://localhost:8000/index",
        json={
            "provider": "openai",
            "model": "text-embedding-3-small",
            "index_name": f"test_{db_type}",
            "texts": documents,
            "db_type": db_type
        }
    )
    print(f"Index créé avec {db_type}")

# Requêtes de test
test_queries = [
    "How do computers learn from data?",
    "What is the relationship between AI and neural networks?",
    "How do machines understand images?",
    "What are different learning approaches in AI?"
]

# Variables pour collecter les résultats
results = {db_type: {"latency": [], "first_result": []} for db_type in db_types}

# Effectuer les tests
for query in test_queries:
    print(f"\nTest de la requête: '{query}'")
    
    for db_type in db_types:
        # Mesurer le temps de réponse
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/search",
            json={
                "provider": "openai",
                "model": "text-embedding-3-small",
                "index_name": f"test_{db_type}",
                "query": query,
                "top_k": 3,
                "db_type": db_type
            }
        )
        latency = time.time() - start_time
        
        # Collecter les résultats
        search_results = response.json()["results"]
        first_result = search_results[0]["text"] if search_results else "Aucun résultat"
        
        # Stocker les mesures
        results[db_type]["latency"].append(latency)
        results[db_type]["first_result"].append(first_result)
        
        print(f"  {db_type}: {latency:.4f}s - Premier résultat: '{first_result[:50]}...'")

# Analyser les résultats
performance_data = []
for db_type in db_types:
    avg_latency = np.mean(results[db_type]["latency"])
    performance_data.append([
        db_type,
        f"{avg_latency:.4f}s",
        f"{min(results[db_type]['latency']):.4f}s",
        f"{max(results[db_type]['latency']):.4f}s"
    ])

# Afficher les résultats dans un tableau
print("\n=== RÉSULTATS DE PERFORMANCE ===")
print(tabulate(
    performance_data,
    headers=["Base de données", "Latence moyenne", "Latence min", "Latence max"],
    tablefmt="grid"
))

# Nettoyage - supprimer les index de test
for db_type in db_types:
    requests.delete(f"http://localhost:8000/index/test_{db_type}?db_type={db_type}")
    print(f"Index test_{db_type} supprimé")
```

---

## 📚 Guide des fonctionnalités avancées

### 🧮 Comparaison directe de textes

Calculez rapidement la similarité entre deux textes sans créer d'index:

```bash
curl -X POST "http://localhost:8000/compare?texts=Artificial%20intelligence%20is%20revolutionizing%20industries.&texts=AI%20is%20changing%20how%20businesses%20operate.&provider=openai&model=text-embedding-3-small"
```

Réponse:
```json
{
  "text1": "Artificial intelligence is revolutionizing industries.",
  "text2": "AI is changing how businesses operate.",
  "similarity": 0.8712,
  "provider": "openai",
  "model": "text-embedding-3-small"
}
```

### 📋 Listing des index disponibles

```bash
curl -X GET http://localhost:8000/indexes
```

Réponse:
```json
[
  {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "index_name": "ai_concepts",
    "total_chunks": 8,
    "created_at": "2023-08-15T14:23:45.123456",
    "updated_at": "2023-08-16T09:12:34.567890",
    "dimension": 1536,
    "db_type": "faiss"
  },
  {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "index_name": "ai_concepts_chroma",
    "total_chunks": 5,
    "created_at": "2023-08-15T15:45:12.345678",
    "dimension": 1536,
    "db_type": "chroma"
  },
  {
    "provider": "cohere",
    "model": "embed-english-v3.0",
    "index_name": "ai_concepts_lance",
    "total_chunks": 5,
    "created_at": "2023-08-15T16:30:22.345678",
    "dimension": 1024,
    "db_type": "lancedb"
  }
]
```

### 🔎 Obtenir les informations sur un index spécifique

```bash
curl -X GET "http://localhost:8000/index/ai_concepts?db_type=faiss"
```

Réponse:
```json
{
  "provider": "openai",
  "model": "text-embedding-3-small",
  "index_name": "ai_concepts",
  "total_chunks": 8,
  "created_at": "2023-08-15T14:23:45.123456",
  "updated_at": "2023-08-16T09:12:34.567890",
  "dimension": 1536,
  "db_type": "faiss"
}
```

---

## 🔒 Sécurité et bonnes pratiques

- 🔑 **Gestion des clés**: Stockées localement dans `.env`, jamais exposées
- 🛡️ **CORS**: Configuration sécurisée pour les requêtes cross-origin
- 📝 **Logging**: Détails utiles sans informations sensibles
- 🔄 **Rate Limiting**: Respect des limites de taux des fournisseurs
- 💾 **Stockage local**: Les données restent sur votre serveur, pas d'envoi externe
- 🧹 **Nettoyage facile**: Suppression d'index via API simple

---

## 🤝 Contribution

Les contributions sont les bienvenues! Voici comment participer:

1. 🍴 **Fork** le dépôt
2. 🔄 Créez une **branche** (`git checkout -b feature/ma-fonctionnalite`)
3. ✏️ Faites vos **modifications**
4. 📦 **Commit** vos changements (`git commit -m 'Ajout de ma fonctionnalité'`)
5. 📤 **Push** vers la branche (`git push origin feature/ma-fonctionnalite`)
6. 🔍 Ouvrez une **Pull Request**

### 💼 Idées de contributions

- 🧪 Support de nouveaux fournisseurs d'embeddings
- 💾 Intégration d'autres bases de données vectorielles (PGVector, Qdrant, Weaviate...)
- 📝 Amélioration de la documentation
- ✨ Fonctionnalités avancées (clustering, chunking automatique)
- 🚀 Optimisations de performance
- 🌐 Support pour la multimodalité (embeddings d'images, audio, etc.)

---

## 📜 Licence

Ce projet est sous licence [MIT](LICENSE) - voir le fichier LICENSE pour plus de détails.

---

## ❓ FAQ

### 🔄 Quelle base de données vectorielle choisir?
- **FAISS** pour performance pure et grands volumes de données
- **ChromaDB** pour applications RAG et intégration IA avancée
- **LanceDB** pour persistance, requêtes complexes et données structurées
- **Cosine** pour prototypage et petits ensembles de données

### 🔍 Quelle est la meilleure dimension pour les embeddings?
En général, plus la dimension est élevée, plus la précision est grande, mais au prix de plus de ressources. 1536 est un bon compromis, 3072 pour une précision maximale.

### 🧩 Comment chunker mes documents avant de les indexer?
Utilisez une bibliothèque comme LangChain ou LlamaIndex pour découper vos documents avant de les envoyer à l'API.

### 💰 Comment optimiser les coûts des API d'embeddings?
Générez les embeddings une seule fois et stockez-les. Utilisez des modèles plus légers pour les cas d'usage moins critiques.

### 🔧 Comment puis-je améliorer la précision de ma recherche?
Expérimentez avec différents modèles, ajustez la taille des chunks, et utilisez des techniques de query expansion.

### 📦 Comment installer les bases de données optionnelles?
Installez ChromaDB avec `pip install chromadb` et LanceDB avec `pip install lancedb`.

---

## 👨‍💻 Auteurs

- 🚀 [Simon-Pierre Boucher](https://github.com/simonpierreboucher0) - Créateur principal

---

<p align="center">
⭐ N'oubliez pas de mettre une étoile si ce projet vous a été utile! ⭐
</p>

<p align="center">
🔗 <a href="https://github.com/simonpierreboucher0/embedding-api">GitHub</a> | 
🐛 <a href="https://github.com/simonpierreboucher0/embedding-api/issues">Signaler un problème</a> | 
💡 <a href="https://github.com/simonpierreboucher0/embedding-api/discussions">Discussions</a>
</p>
