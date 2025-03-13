# 🌐 Embedding API Gateway 🚀

## 🔥 Passerelle universelle pour tous vos services d'embeddings et recherche sémantique 🔥

[![GitHub stars](https://img.shields.io/github/stars/simonpierreboucher0/embedding-api?style=social)](https://github.com/simonpierreboucher0/embedding-api/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.105.0+-green.svg)](https://fastapi.tiangolo.com/)

---

## ✨ Caractéristiques principales

🔄 **Interface unifiée** - Une API pour tous les fournisseurs d'embeddings  
🧩 **Double moteur de recherche** - FAISS ou Similarité Cosinus au choix  
🔍 **Recherche vectorielle** - Recherche sémantique ultra-rapide avec FAISS  
📊 **Stockage flexible** - Formats optimisés pour rapidité ou lisibilité  
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
uvicorn app:app --reload --host 0.0.0.0 --port 8001
```

### 📋 Obtenir la liste des modèles
```bash
curl -X GET http://localhost:8001/models
```

### 🧠 Générer des embeddings
```bash
curl -X POST http://localhost:8001/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "texts": "Voici un exemple de texte pour générer un embedding."
  }'
```

### 🗄️ Créer un index FAISS
```bash
curl -X POST http://localhost:8001/index \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "index_name": "mon_index_articles",
    "texts": [
      "L'intelligence artificielle révolutionne notre monde.",
      "Le machine learning permet d'automatiser des tâches complexes.",
      "La recherche sémantique utilise des vecteurs d'embedding.",
      "FAISS est une bibliothèque efficace pour la recherche vectorielle.",
      "Les LLMs utilisent des transformers pour comprendre le contexte."
    ],
    "method": "faiss"
  }'
```

### 📊 Créer un index avec similarité cosinus
```bash
curl -X POST http://localhost:8001/index \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "index_name": "mon_index_json",
    "texts": [
      "L'intelligence artificielle révolutionne notre monde.",
      "Le machine learning permet d'automatiser des tâches complexes.",
      "La recherche sémantique utilise des vecteurs d'embedding.",
      "FAISS est une bibliothèque efficace pour la recherche vectorielle.",
      "Les LLMs utilisent des transformers pour comprendre le contexte."
    ],
    "method": "cosine"
  }'
```

### 🔍 Recherche sémantique
```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "index_name": "mon_index_articles",
    "query": "Comment fonctionne l'IA?",
    "top_k": 3,
    "method": "faiss"
  }'
```

### 🔄 Mettre à jour un index existant
```bash
curl -X PUT http://localhost:8001/index/mon_index_articles \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "texts": [
      "Les réseaux de neurones sont inspirés du cerveau humain.",
      "Le deep learning est une sous-catégorie du machine learning.",
      "Les embeddings permettent de capturer la sémantique du texte."
    ],
    "method": "faiss"
  }'
```

### 🧮 Comparer deux textes directement
```bash
curl -X POST "http://localhost:8001/compare?texts=L'IA%20est%20revolutionnaire&texts=L'intelligence%20artificielle%20transforme%20le%20monde&provider=openai&model=text-embedding-ada-002" \
  -H "accept: application/json"
```

---

## 📊 Structure complète des requêtes et réponses

### 📝 Génération d'embeddings

**Requête:**
```json
{
  "provider": "openai",              // 🌐 Fournisseur (obligatoire)
  "model": "text-embedding-ada-002", // 🤖 Modèle spécifique (obligatoire)
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
  "model": "text-embedding-ada-002", // 🤖 Modèle utilisé
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
  "model": "text-embedding-ada-002", // 🤖 Modèle spécifique (obligatoire)
  "index_name": "mon_index",         // 📚 Nom de l'index (obligatoire)
  "texts": [                         // 📄 Textes à indexer (obligatoire)
    "Premier document à indexer",
    "Deuxième document à indexer",
    "Troisième document à indexer"
  ],
  "method": "faiss",                 // 🔍 Méthode d'indexation: "faiss" ou "cosine"
  "encoding_format": "float",        // 📊 Format d'encodage (optionnel)
  "input_type": "classification"     // 🏷️ Type d'entrée (pour Cohere, optionnel)
}
```

**Réponse:**
```json
{
  "provider": "openai",              // 🌐 Fournisseur utilisé
  "model": "text-embedding-ada-002", // 🤖 Modèle utilisé
  "index_name": "mon_index",         // 📚 Nom de l'index créé
  "total_chunks": 3,                 // 🧩 Nombre de documents indexés
  "created_at": "2023-08-15T14:23:45.123456", // ⏰ Date de création
  "dimension": 1536,                 // 📏 Dimension des vecteurs
  "method": "faiss"                  // 🔍 Méthode d'indexation utilisée
}
```

### 🔍 Recherche sémantique

**Requête:**
```json
{
  "provider": "openai",              // 🌐 Fournisseur (obligatoire)
  "model": "text-embedding-ada-002", // 🤖 Modèle spécifique (obligatoire)
  "index_name": "mon_index",         // 📚 Nom de l'index (obligatoire)
  "query": "Ma requête de recherche", // 🔎 Texte de la requête (obligatoire)
  "top_k": 5,                        // 🔝 Nombre de résultats souhaités
  "method": "faiss"                  // 🔍 Méthode de recherche: "faiss" ou "cosine"
}
```

**Réponse:**
```json
{
  "index_name": "mon_index",         // 📚 Nom de l'index utilisé
  "provider": "openai",              // 🌐 Fournisseur utilisé
  "model": "text-embedding-ada-002", // 🤖 Modèle utilisé
  "query": "Ma requête de recherche", // 🔎 Texte de la requête
  "method": "faiss",                 // 🔍 Méthode utilisée
  "results": [                       // 📋 Résultats de recherche
    {
      "chunk_id": 2,                 // 🆔 ID du document
      "text": "Deuxième document à indexer", // 📄 Texte du document
      "distance": 0.125,             // 📏 Distance (FAISS uniquement)
      "similarity": 0.89,            // 📊 Score de similarité
      "rank": 1                      // 🏅 Rang dans les résultats
    },
    // ... autres résultats
  ]
}
```

---

## 🧪 Comparaison des méthodes de recherche

### 🚀 FAISS (Fast Library for Approximate Nearest Neighbors)

**Avantages:**
- ⚡ **Ultra rapide** pour les grands ensembles de données
- 🔍 **Recherche approximative** optimisée pour hautes dimensions
- 📈 **Passage à l'échelle** efficace avec des millions de vecteurs
- 🧠 **Optimisations mémoire** pour les grandes collections

**Utilisez FAISS quand:**
- 🗄️ Vous avez de grandes collections de documents (>10K textes)
- ⏱️ La vitesse de recherche est critique
- 💽 Vous avez des contraintes de mémoire pour de grands index

### 📊 Similarité Cosinus (avec stockage JSON)

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

# Créer l'index
requests.post(
    "http://localhost:8001/index",
    json={
        "provider": "openai",
        "model": "text-embedding-ada-002",
        "index_name": "docs_techniques",
        "texts": documents,
        "method": "faiss"
    }
)

# 2. Quand l'utilisateur pose une question
user_query = "Comment fonctionne la recherche vectorielle?"

# 3. Rechercher les documents pertinents
search_response = requests.post(
    "http://localhost:8001/search",
    json={
        "provider": "openai",
        "model": "text-embedding-ada-002",
        "index_name": "docs_techniques",
        "query": user_query,
        "top_k": 2,
        "method": "faiss"
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

### 🧪 Test A/B de différents modèles d'embeddings

```python
import requests
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Liste de modèles à comparer
models_to_test = [
    {"provider": "openai", "model": "text-embedding-ada-002"},
    {"provider": "openai", "model": "text-embedding-3-small"},
    {"provider": "cohere", "model": "embed-english-v3.0"},
    {"provider": "mistral", "model": "mistral-embed"}
]

# Ensemble de test (requêtes et documents pertinents attendus)
test_queries = [
    {
        "query": "Comment fonctionne l'apprentissage profond?",
        "relevant_docs": [0, 2, 5]  # Indices des documents pertinents
    },
    {
        "query": "Qu'est-ce que le traitement du langage naturel?",
        "relevant_docs": [1, 4, 7]
    }
    # Ajoutez plus de requêtes...
]

documents = [
    "Le deep learning est un sous-domaine du machine learning qui utilise des réseaux de neurones.",
    "Le NLP ou traitement du langage naturel permet aux machines de comprendre le texte.",
    "Les réseaux de neurones profonds contiennent plusieurs couches cachées.",
    "Python est un langage de programmation populaire pour l'IA.",
    "BERT est un modèle de langage pré-entraîné pour le NLP.",
    "L'apprentissage profond nécessite généralement de grandes quantités de données.",
    "TensorFlow et PyTorch sont des frameworks populaires pour le deep learning.",
    "Les modèles de langue comme GPT utilisent le NLP pour générer du texte.",
    "Les embeddings vectoriels sont essentiels pour la recherche sémantique."
]

results = {}

# Tester chaque modèle
for model_info in models_to_test:
    provider = model_info["provider"]
    model = model_info["model"]
    
    # Créer un index avec ce modèle
    index_name = f"test_{provider}_{model.replace('-', '_').replace('/', '_')}"
    
    requests.post(
        "http://localhost:8001/index",
        json={
            "provider": provider,
            "model": model,
            "index_name": index_name,
            "texts": documents,
            "method": "faiss"
        }
    )
    
    # Évaluer sur chaque requête
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for test_case in test_queries:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        
        # Faire la recherche
        search_response = requests.post(
            "http://localhost:8001/search",
            json={
                "provider": provider,
                "model": model,
                "index_name": index_name,
                "query": query,
                "top_k": len(documents),
                "method": "faiss"
            }
        )
        
        # Récupérer les résultats
        results_data = search_response.json()["results"]
        retrieved_indices = [result["chunk_id"] for result in results_data]
        
        # Créer un vecteur de pertinence binaire pour l'évaluation
        y_true = np.zeros(len(documents))
        y_true[relevant_docs] = 1
        
        y_pred = np.zeros(len(documents))
        y_pred[retrieved_indices[:5]] = 1  # Considérer les 5 premiers comme pertinents
        
        # Calculer les métriques
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # Stocker les résultats pour ce modèle
    results[f"{provider}/{model}"] = {
        "precision_avg": np.mean(precision_scores),
        "recall_avg": np.mean(recall_scores),
        "f1_avg": np.mean(f1_scores)
    }
    
    # Nettoyer l'index après le test
    requests.delete(f"http://localhost:8001/index/{index_name}?method=faiss")

# Afficher les résultats comparatifs
print("\n=== RÉSULTATS DE LA COMPARAISON DES MODÈLES ===")
for model_name, metrics in results.items():
    print(f"📊 {model_name}:")
    print(f"  Précision: {metrics['precision_avg']:.4f}")
    print(f"  Rappel: {metrics['recall_avg']:.4f}")
    print(f"  F1-Score: {metrics['f1_avg']:.4f}")
    print("---")
```

---

## 📚 Guide des fonctionnalités avancées

### 🧮 Comparaison directe de textes

Calculez rapidement la similarité entre deux textes sans créer d'index:

```bash
curl -X POST "http://localhost:8001/compare?texts=L'IA%20est%20revolutionnaire&texts=L'intelligence%20artificielle%20transforme%20le%20monde&provider=openai&model=text-embedding-ada-002"
```

Réponse:
```json
{
  "text1": "L'IA est revolutionnaire",
  "text2": "L'intelligence artificielle transforme le monde",
  "similarity": 0.8712,
  "provider": "openai",
  "model": "text-embedding-ada-002"
}
```

### 📋 Listing des index disponibles

```bash
curl -X GET http://localhost:8001/indexes
```

Réponse:
```json
[
  {
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "index_name": "mon_index_articles",
    "total_chunks": 8,
    "created_at": "2023-08-15T14:23:45.123456",
    "updated_at": "2023-08-16T09:12:34.567890",
    "dimension": 1536,
    "method": "faiss"
  },
  {
    "provider": "cohere",
    "model": "embed-english-v3.0",
    "index_name": "mon_index_json",
    "total_chunks": 5,
    "created_at": "2023-08-15T15:45:12.345678",
    "dimension": 1024,
    "method": "cosine"
  }
]
```

### 🔎 Obtenir les informations sur un index spécifique

```bash
curl -X GET "http://localhost:8001/index/mon_index_articles?method=faiss"
```

Réponse:
```json
{
  "provider": "openai",
  "model": "text-embedding-ada-002",
  "index_name": "mon_index_articles",
  "total_chunks": 8,
  "created_at": "2023-08-15T14:23:45.123456",
  "updated_at": "2023-08-16T09:12:34.567890",
  "dimension": 1536,
  "method": "faiss"
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
- 📝 Amélioration de la documentation
- ✨ Fonctionnalités avancées (clustering, chunking automatique)
- 🚀 Optimisations de performance
- 🌐 Support de la persistance dans des bases de données vectorielles

---

## 📜 Licence

Ce projet est sous licence [MIT](LICENSE) - voir le fichier LICENSE pour plus de détails.

---

## ❓ FAQ

### 🔄 Quelle méthode de recherche choisir?
FAISS pour grands datasets et performance, Cosinus pour petits datasets et transparence.

### 🔍 Quelle est la meilleure dimension pour les embeddings?
En général, plus la dimension est élevée, plus la précision est grande, mais au prix de plus de ressources. 1536 est un bon compromis.

### 🧩 Comment chunker mes documents avant de les indexer?
Utilisez une bibliothèque comme LangChain ou LlamaIndex pour découper vos documents avant de les envoyer à l'API.

### 💰 Comment optimiser les coûts des API d'embeddings?
Générez les embeddings une seule fois et stockez-les. Utilisez des modèles plus légers pour les cas d'usage moins critiques.

### 🔧 Comment puis-je améliorer la précision de ma recherche?
Expérimentez avec différents modèles, ajustez la taille des chunks, et utilisez des techniques de query expansion.

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
