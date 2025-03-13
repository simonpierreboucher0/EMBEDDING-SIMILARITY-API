import os
import json
import logging
import faiss
import numpy as np
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Literal
from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
from dotenv import load_dotenv
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Dépendances conditionnelles pour les différentes bases de données vectorielles
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

# Charger les variables d'environnement depuis un fichier .env
load_dotenv()

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding API Gateway", description="API Gateway unifié pour différents modèles d'embeddings et recherche sémantique avec Chroma, LanceDB ou FAISS")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurations de base - créer les répertoires nécessaires
DATA_DIRECTORY = "vector_data"
os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(f"{DATA_DIRECTORY}/faiss", exist_ok=True)
os.makedirs(f"{DATA_DIRECTORY}/chroma", exist_ok=True)
os.makedirs(f"{DATA_DIRECTORY}/lancedb", exist_ok=True)
os.makedirs(f"{DATA_DIRECTORY}/json", exist_ok=True)

# Enumération des providers disponibles
class Provider(str, Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    MISTRAL = "mistral"
    DEEPINFRA = "deepinfra"
    VOYAGE = "voyage"

# Enumération des méthodes de recherche
class VectorDBType(str, Enum):
    FAISS = "faiss"
    CHROMA = "chroma"
    LANCEDB = "lancedb"
    COSINE = "cosine"  # Gardé pour compatibilité

# Modèles disponibles par provider
MODELS = {
    Provider.OPENAI: [
        "text-embedding-ada-002", 
        "text-embedding-3-small",
        "text-embedding-3-large"
    ],
    Provider.COHERE: [
        "embed-english-v3.0",
        "embed-multilingual-v3.0",
        "embed-english-light-v3.0",
        "embed-multilingual-light-v3.0"
    ],
    Provider.MISTRAL: [
        "mistral-embed"
    ],
    Provider.DEEPINFRA: [
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "thenlper/gte-large",
        "thenlper/gte-base",
        "intfloat/e5-large-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ],
    Provider.VOYAGE: [
        "voyage-large-2",
        "voyage-large-2-instruct",
        "voyage-code-2"
    ]
}

# Classe pour gérer les API keys depuis .env
class APIKeys:
    def __init__(self):
        self.keys = {
            Provider.OPENAI: os.getenv("OPENAI_API_KEY"),
            Provider.COHERE: os.getenv("COHERE_API_KEY"),
            Provider.MISTRAL: os.getenv("MISTRAL_API_KEY"),
            Provider.DEEPINFRA: os.getenv("DEEPINFRA_API_KEY"),
            Provider.VOYAGE: os.getenv("VOYAGE_API_KEY")
        }

    def get_key(self, provider: Provider):
        key = self.keys.get(provider)
        if not key:
            raise HTTPException(status_code=401, detail=f"API key for {provider} not found")
        return key

api_keys = APIKeys()

# Modèles de données pour l'API
class EmbeddingRequest(BaseModel):
    provider: Provider
    model: str
    texts: Union[str, List[str]]
    encoding_format: Optional[str] = "float"  # Pour OpenAI, Mistral, DeepInfra
    input_type: Optional[str] = "classification"  # Pour Cohere

class SearchRequest(BaseModel):
    provider: Provider
    model: str
    index_name: str
    query: str
    top_k: int = 5
    db_type: VectorDBType = VectorDBType.FAISS

class CreateIndexRequest(BaseModel):
    provider: Provider
    model: str
    index_name: str
    texts: List[str]
    encoding_format: Optional[str] = "float"
    input_type: Optional[str] = "classification"  # Pour Cohere
    db_type: VectorDBType = VectorDBType.FAISS

class UpdateIndexRequest(BaseModel):
    provider: Provider
    model: str
    index_name: str
    texts: List[str]
    db_type: VectorDBType = VectorDBType.FAISS

class EmbeddingResult(BaseModel):
    embeddings: List[List[float]]
    model: str
    provider: str
    dimension: int
    total_tokens: Optional[int] = None

class SearchResult(BaseModel):
    index_name: str
    provider: str
    model: str
    query: str
    db_type: VectorDBType
    results: List[Dict[str, Any]]

class IndexInfo(BaseModel):
    provider: str
    model: str
    index_name: str
    total_chunks: int
    created_at: str
    updated_at: Optional[str] = None
    dimension: int
    db_type: VectorDBType

# Fonctions utilitaires pour les embeddings
def get_openai_embedding(texts, model, encoding_format, api_key):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": encoding_format
    }

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"OpenAI API error: {response.text}")
    
    result = response.json()
    embeddings = [data["embedding"] for data in result["data"]]
    
    return {
        "embeddings": embeddings,
        "total_tokens": result.get("usage", {}).get("total_tokens", 0)
    }

def get_cohere_embedding(texts, model, input_type, api_key):
    url = "https://api.cohere.com/v2/embed"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "texts": texts if isinstance(texts, list) else [texts],
        "input_type": input_type,
        "embedding_types": ["float"]
    }

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Cohere API error: {response.text}")
    
    result = response.json()
    return {
        "embeddings": result["embeddings"]["float"]
    }

def get_mistral_embedding(texts, model, encoding_format, api_key):
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Normaliser en liste
    if isinstance(texts, str):
        texts = [texts]
    
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": encoding_format
    }

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Mistral API error: {response.text}")
    
    result = response.json()
    return {
        "embeddings": [data["embedding"] for data in result["data"]]
    }

def get_deepinfra_embedding(texts, model, encoding_format, api_key):
    url = "https://api.deepinfra.com/v1/openai/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": encoding_format
    }

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"DeepInfra API error: {response.text}")
    
    result = response.json()
    return {
        "embeddings": [data["embedding"] for data in result["data"]]
    }

def get_voyage_embedding(texts, model, api_key):
    url = "https://api.voyageai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Normaliser en liste
    if isinstance(texts, str):
        texts = [texts]
    
    payload = {
        "model": model,
        "input": texts
    }

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Voyage API error: {response.text}")
    
    result = response.json()
    return {
        "embeddings": [data["embedding"] for data in result["data"]]
    }

# Fonction pour obtenir des embeddings selon le provider
def get_embeddings_by_provider(provider, model, texts, **kwargs):
    """Fonction générique pour obtenir des embeddings selon le provider"""
    if provider == Provider.OPENAI:
        encoding_format = kwargs.get('encoding_format', 'float')
        return get_openai_embedding(
            texts, 
            model, 
            encoding_format, 
            api_keys.get_key(Provider.OPENAI)
        )
            
    elif provider == Provider.COHERE:
        input_type = kwargs.get('input_type', 'classification')
        return get_cohere_embedding(
            texts, 
            model, 
            input_type, 
            api_keys.get_key(Provider.COHERE)
        )
            
    elif provider == Provider.MISTRAL:
        encoding_format = kwargs.get('encoding_format', 'float')
        return get_mistral_embedding(
            texts, 
            model, 
            encoding_format, 
            api_keys.get_key(Provider.MISTRAL)
        )
            
    elif provider == Provider.DEEPINFRA:
        encoding_format = kwargs.get('encoding_format', 'float')
        return get_deepinfra_embedding(
            texts, 
            model, 
            encoding_format, 
            api_keys.get_key(Provider.DEEPINFRA)
        )
            
    elif provider == Provider.VOYAGE:
        return get_voyage_embedding(
            texts, 
            model, 
            api_keys.get_key(Provider.VOYAGE)
        )
            
    else:
        raise HTTPException(status_code=400, detail=f"Provider {provider} not supported")

# Fonctions de gestion des index FAISS
def create_faiss_index(provider, model, index_name, texts, embeddings, **kwargs):
    """Créer un index FAISS à partir d'embeddings"""
    # Convertir les embeddings en numpy array
    embeddings_np = np.array(embeddings, dtype=np.float32)
    
    # Dimension des embeddings
    embedding_dim = embeddings_np.shape[1]
    
    # Créer l'index FAISS
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)
    
    # Métadonnées des chunks
    chunks_metadata = {
        "created_at": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "total_chunks": len(texts),
        "embedding_dim": embedding_dim,
        "db_type": "faiss",
        **kwargs,  # Autres paramètres spécifiques au provider
        "chunks": [
            {
                "id": i,
                "text": text,
                "embedding_index": i
            }
            for i, text in enumerate(texts)
        ]
    }
    
    # Sauvegarder l'index et les métadonnées
    faiss.write_index(index, f"{DATA_DIRECTORY}/faiss/{index_name}.faiss")
    
    with open(f"{DATA_DIRECTORY}/faiss/{index_name}.json", 'w') as f:
        json.dump(chunks_metadata, f, indent=2)
    
    return index, chunks_metadata

def update_faiss_index(provider, model, index_name, texts, new_embeddings, **kwargs):
    """Mettre à jour un index FAISS existant"""
    try:
        # Charger l'index existant
        index = faiss.read_index(f"{DATA_DIRECTORY}/faiss/{index_name}.faiss")
        
        # Charger les métadonnées
        with open(f"{DATA_DIRECTORY}/faiss/{index_name}.json", 'r') as f:
            chunks_metadata = json.load(f)
        
        # Obtenir les embeddings existants
        ntotal = index.ntotal
        dimension = index.d
        
        # Convertir les nouveaux embeddings en numpy array
        new_embeddings_np = np.array(new_embeddings, dtype=np.float32)
        
        # Vérifier la compatibilité dimensionnelle
        if dimension != new_embeddings_np.shape[1]:
            raise HTTPException(
                status_code=400, 
                detail=f"Dimension mismatch: existing {dimension}, new {new_embeddings_np.shape[1]}"
            )
        
        # Ajouter les nouveaux embeddings à l'index
        index.add(new_embeddings_np)
        
        # Mettre à jour les métadonnées
        start_id = len(chunks_metadata["chunks"])
        new_chunks = [
            {
                "id": start_id + i,
                "text": text,
                "embedding_index": ntotal + i
            }
            for i, text in enumerate(texts)
        ]
        chunks_metadata["chunks"].extend(new_chunks)
        chunks_metadata["total_chunks"] = len(chunks_metadata["chunks"])
        chunks_metadata["updated_at"] = datetime.now().isoformat()
        
        # Sauvegarder l'index et les métadonnées
        faiss.write_index(index, f"{DATA_DIRECTORY}/faiss/{index_name}.faiss")
        
        with open(f"{DATA_DIRECTORY}/faiss/{index_name}.json", 'w') as f:
            json.dump(chunks_metadata, f, indent=2)
        
        return index, chunks_metadata
        
    except FileNotFoundError:
        # Si l'index n'existe pas, le créer
        return create_faiss_index(provider, model, index_name, texts, new_embeddings, **kwargs)

def search_faiss_index(index_name, query_embedding, k=5):
    """Rechercher dans un index FAISS"""
    try:
        # Charger l'index
        index = faiss.read_index(f"{DATA_DIRECTORY}/faiss/{index_name}.faiss")
        
        # Charger les métadonnées
        with open(f"{DATA_DIRECTORY}/faiss/{index_name}.json", 'r') as f:
            chunks_metadata = json.load(f)
        
        # Préparer l'embedding de requête
        query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Effectuer la recherche
        distances, indices = index.search(query_embedding_np, k)
        
        # Formater les résultats
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(chunks_metadata["chunks"]):  # Vérification de sécurité
                # Trouver le chunk correspondant à l'indice d'embedding
                chunk = None
                for c in chunks_metadata["chunks"]:
                    if c["embedding_index"] == idx:
                        chunk = c
                        break
                
                if chunk:
                    results.append({
                        "chunk_id": chunk["id"],
                        "text": chunk["text"],
                        "distance": float(dist),
                        "score": 1 / (1 + float(dist)),  # Convertir distance en score de similarité
                        "rank": i + 1
                    })
        
        return results
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"FAISS Index '{index_name}' not found")

# Fonctions pour ChromaDB
def create_chroma_index(provider, model, index_name, texts, embeddings, **kwargs):
    """Créer un index avec ChromaDB"""
    if not CHROMA_AVAILABLE:
        raise HTTPException(status_code=400, detail="ChromaDB not installed. Please install with 'pip install chromadb'")
    
    try:
        # Créer un client ChromaDB persistant
        client = chromadb.PersistentClient(path=f"{DATA_DIRECTORY}/chroma")
        
        # Créer une collection
        collection = client.create_collection(
            name=index_name,
            metadata={
                "provider": provider,
                "model": model,
                "created_at": datetime.now().isoformat(),
                "embedding_dim": len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0,
                **kwargs
            }
        )
        
        # Ajouter les documents avec leurs embeddings
        ids = [str(i) for i in range(len(texts))]
        
        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"text": text, "source": f"{provider}/{model}"} for text in texts]
        )
        
        # Métadonnées pour notre API
        metadata = {
            "created_at": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "total_chunks": len(texts),
            "embedding_dim": len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0,
            "db_type": "chroma"
        }
        
        # Sauvegarder les métadonnées dans notre format
        with open(f"{DATA_DIRECTORY}/chroma/{index_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return collection, metadata
        
    except Exception as e:
        logger.error(f"Error creating ChromaDB index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ChromaDB error: {str(e)}")

def update_chroma_index(provider, model, index_name, texts, new_embeddings, **kwargs):
    """Mettre à jour un index ChromaDB existant"""
    if not CHROMA_AVAILABLE:
        raise HTTPException(status_code=400, detail="ChromaDB not installed. Please install with 'pip install chromadb'")
    
    try:
        # Créer un client ChromaDB persistant
        client = chromadb.PersistentClient(path=f"{DATA_DIRECTORY}/chroma")
        
        # Vérifier si la collection existe
        try:
            collection = client.get_collection(name=index_name)
        except:
            # Si la collection n'existe pas, la créer
            return create_chroma_index(provider, model, index_name, texts, new_embeddings, **kwargs)
        
        # Charger les métadonnées existantes
        try:
            with open(f"{DATA_DIRECTORY}/chroma/{index_name}_metadata.json", 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {
                "created_at": datetime.now().isoformat(),
                "provider": provider,
                "model": model,
                "total_chunks": 0,
                "embedding_dim": len(new_embeddings[0]) if new_embeddings and len(new_embeddings) > 0 else 0,
                "db_type": "chroma"
            }
        
        # Obtenir le nombre actuel de documents
        current_count = metadata.get("total_chunks", 0)
        
        # Générer de nouveaux IDs
        ids = [str(i + current_count) for i in range(len(texts))]
        
        # Ajouter les nouveaux documents
        collection.add(
            documents=texts,
            embeddings=new_embeddings,
            ids=ids,
            metadatas=[{"text": text, "source": f"{provider}/{model}"} for text in texts]
        )
        
        # Mettre à jour les métadonnées
        metadata["total_chunks"] = current_count + len(texts)
        metadata["updated_at"] = datetime.now().isoformat()
        
        # Sauvegarder les métadonnées mises à jour
        with open(f"{DATA_DIRECTORY}/chroma/{index_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return collection, metadata
        
    except Exception as e:
        logger.error(f"Error updating ChromaDB index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ChromaDB error: {str(e)}")

def search_chroma_index(index_name, query_embedding, k=5):
    """Rechercher dans un index ChromaDB"""
    if not CHROMA_AVAILABLE:
        raise HTTPException(status_code=400, detail="ChromaDB not installed. Please install with 'pip install chromadb'")
    
    try:
        # Créer un client ChromaDB persistant
        client = chromadb.PersistentClient(path=f"{DATA_DIRECTORY}/chroma")
        
        # Obtenir la collection
        try:
            collection = client.get_collection(name=index_name)
        except:
            raise HTTPException(status_code=404, detail=f"ChromaDB collection '{index_name}' not found")
        
        # Effectuer la recherche par vecteur
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Formater les résultats
        formatted_results = []
        for i, (document, dist, id_val) in enumerate(zip(results["documents"][0], results["distances"][0], results["ids"][0])):
            formatted_results.append({
                "chunk_id": id_val,
                "text": document,
                "distance": float(dist),
                "score": 1 - (float(dist) / 2),  # ChromaDB utilise la distance cosinus, convertir en score
                "rank": i + 1
            })
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error searching ChromaDB index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ChromaDB error: {str(e)}")

# Fonctions pour LanceDB
def create_lancedb_index(provider, model, index_name, texts, embeddings, **kwargs):
    """Créer un index avec LanceDB"""
    if not LANCEDB_AVAILABLE:
        raise HTTPException(status_code=400, detail="LanceDB not installed. Please install with 'pip install lancedb'")
    
    try:
        # Créer une connexion à la base de données
        db = lancedb.connect(f"{DATA_DIRECTORY}/lancedb")
        
        # Préparer les données
        data = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            data.append({
                "id": i,
                "text": text,
                "vector": embedding
            })
        
        # Créer la table
        table = db.create_table(
            index_name,
            data=data,
            mode="overwrite"
        )
        
        # Métadonnées pour notre API
        metadata = {
            "created_at": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "total_chunks": len(texts),
            "embedding_dim": len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0,
            "db_type": "lancedb"
        }
        
        # Sauvegarder les métadonnées dans notre format
        with open(f"{DATA_DIRECTORY}/lancedb/{index_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return table, metadata
        
    except Exception as e:
        logger.error(f"Error creating LanceDB index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LanceDB error: {str(e)}")

def update_lancedb_index(provider, model, index_name, texts, new_embeddings, **kwargs):
    """Mettre à jour un index LanceDB existant"""
    if not LANCEDB_AVAILABLE:
        raise HTTPException(status_code=400, detail="LanceDB not installed. Please install with 'pip install lancedb'")
    
    try:
        # Créer une connexion à la base de données
        db = lancedb.connect(f"{DATA_DIRECTORY}/lancedb")
        
        # Vérifier si la table existe
        try:
            table = db.open_table(index_name)
            
            # Charger les métadonnées existantes
            try:
                with open(f"{DATA_DIRECTORY}/lancedb/{index_name}_metadata.json", 'r') as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                metadata = {
                    "created_at": datetime.now().isoformat(),
                    "provider": provider,
                    "model": model,
                    "total_chunks": 0,
                    "embedding_dim": len(new_embeddings[0]) if new_embeddings and len(new_embeddings) > 0 else 0,
                    "db_type": "lancedb"
                }
            
            # Obtenir le nombre actuel de documents
            current_count = metadata.get("total_chunks", 0)
            
            # Préparer les nouvelles données
            data = []
            for i, (text, embedding) in enumerate(zip(texts, new_embeddings)):
                data.append({
                    "id": current_count + i,
                    "text": text,
                    "vector": embedding
                })
            
            # Ajouter les nouvelles données
            table.add(data=data)
            
            # Mettre à jour les métadonnées
            metadata["total_chunks"] = current_count + len(texts)
            metadata["updated_at"] = datetime.now().isoformat()
            
            # Sauvegarder les métadonnées mises à jour
            with open(f"{DATA_DIRECTORY}/lancedb/{index_name}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return table, metadata
            
        except:
            # Si la table n'existe pas, la créer
            return create_lancedb_index(provider, model, index_name, texts, new_embeddings, **kwargs)
        
    except Exception as e:
        logger.error(f"Error updating LanceDB index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LanceDB error: {str(e)}")

def search_lancedb_index(index_name, query_embedding, k=5):
    """Rechercher dans un index LanceDB"""
    if not LANCEDB_AVAILABLE:
        raise HTTPException(status_code=400, detail="LanceDB not installed. Please install with 'pip install lancedb'")
    
    try:
        # Créer une connexion à la base de données
        db = lancedb.connect(f"{DATA_DIRECTORY}/lancedb")
        
        # Ouvrir la table
        try:
            table = db.open_table(index_name)
        except:
            raise HTTPException(status_code=404, detail=f"LanceDB table '{index_name}' not found")
        
        # Effectuer la recherche
        query_results = table.search(query_embedding).limit(k).to_pandas()
        
        # Afficher les colonnes disponibles dans les logs pour le débogage
        logger.info(f"Available columns in LanceDB result: {query_results.columns.tolist()}")
        
        # Formater les résultats
        formatted_results = []
        for i, row in enumerate(query_results.itertuples()):
            # Essayer de déterminer les colonnes correctes
            try:
                # Si la colonne contenant la distance a différents noms selon les versions
                distance = None
                
                # Essayer différentes colonnes possibles
                for attr_name in ['_distance', '_score', 'score', 'distance']:
                    if hasattr(row, attr_name):
                        distance = float(getattr(row, attr_name))
                        break
                
                # Si aucune colonne de distance n'est trouvée, utiliser l'index comme score
                if distance is None:
                    distance = float(i)
                
                # Trouver la colonne de texte
                text = None
                for attr_name in ['text', 'document', 'content']:
                    if hasattr(row, attr_name):
                        text = getattr(row, attr_name)
                        break
                
                if text is None:
                    text = f"Result {i+1}"
                
                # Trouver l'ID
                id_val = None
                for attr_name in ['id', 'chunk_id', 'Index']:
                    if hasattr(row, attr_name):
                        id_val = getattr(row, attr_name)
                        break
                
                if id_val is None:
                    id_val = i
                
                formatted_results.append({
                    "chunk_id": str(id_val),
                    "text": text,
                    "distance": distance,
                    "score": 1 / (1 + distance),  # Convertir distance en score de similarité
                    "rank": i + 1
                })
            except Exception as e:
                # Si une erreur se produit pour une ligne spécifique, utiliser des valeurs par défaut
                logger.warning(f"Error processing LanceDB result row {i}: {e}")
                formatted_results.append({
                    "chunk_id": str(i),
                    "text": f"Result {i+1}",
                    "distance": float(i),
                    "score": 1 / (1 + float(i)),
                    "rank": i + 1
                })
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error searching LanceDB index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LanceDB error: {str(e)}")

# Fonctions pour la méthode de similarité cosinus
def create_cosine_index(provider, model, index_name, texts, embeddings, **kwargs):
    """Créer un index JSON pour la similarité cosinus"""
    # Dimension des embeddings
    embedding_dim = len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0
    
    # Créer le fichier de données JSON
    index_data = {
        "created_at": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "total_chunks": len(texts),
        "embedding_dim": embedding_dim,
        "db_type": "cosine",
        **kwargs,  # Autres paramètres spécifiques au provider
        "chunks": [
            {
                "id": i,
                "text": text,
                "embedding": embedding
            }
            for i, (text, embedding) in enumerate(zip(texts, embeddings))
        ]
    }
    
    # Sauvegarder les données
    with open(f"{DATA_DIRECTORY}/json/{index_name}.json", 'w') as f:
        json.dump(index_data, f, indent=2)
    
    return index_data

def update_cosine_index(provider, model, index_name, texts, new_embeddings, **kwargs):
    """Mettre à jour un index JSON existant"""
    try:
        # Charger les données existantes
        with open(f"{DATA_DIRECTORY}/json/{index_name}.json", 'r') as f:
            index_data = json.load(f)
        
        # Vérifier la compatibilité dimensionnelle
        existing_dim = index_data["embedding_dim"]
        new_dim = len(new_embeddings[0]) if new_embeddings and len(new_embeddings) > 0 else 0
        
        if existing_dim != new_dim:
            raise HTTPException(
                status_code=400, 
                detail=f"Dimension mismatch: existing {existing_dim}, new {new_dim}"
            )
        
        # Ajouter les nouveaux chunks
        start_id = len(index_data["chunks"])
        new_chunks = [
            {
                "id": start_id + i,
                "text": text,
                "embedding": embedding
            }
            for i, (text, embedding) in enumerate(zip(texts, new_embeddings))
        ]
        index_data["chunks"].extend(new_chunks)
        index_data["total_chunks"] = len(index_data["chunks"])
        index_data["updated_at"] = datetime.now().isoformat()
        
        # Sauvegarder les données mises à jour
        with open(f"{DATA_DIRECTORY}/json/{index_name}.json", 'w') as f:
            json.dump(index_data, f, indent=2)
        
        return index_data
        
    except FileNotFoundError:
        # Si l'index n'existe pas, le créer
        return create_cosine_index(provider, model, index_name, texts, new_embeddings, **kwargs)

def search_cosine_index(index_name, query_embedding, k=5):
    """Rechercher avec la similarité cosinus"""
    try:
        # Charger les données
        with open(f"{DATA_DIRECTORY}/json/{index_name}.json", 'r') as f:
            index_data = json.load(f)
        
        # Extraire les embeddings et les textes
        chunks = index_data["chunks"]
        embeddings = [chunk["embedding"] for chunk in chunks]
        
        # Calculer la similarité cosinus
        query_embedding_np = np.array(query_embedding).reshape(1, -1)
        embeddings_np = np.array(embeddings)
        
        similarities = cosine_similarity(query_embedding_np, embeddings_np)[0]
        
        # Obtenir les indices triés par similarité décroissante
        top_indices = similarities.argsort()[-k:][::-1]
        
        # Formater les résultats
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "chunk_id": chunks[idx]["id"],
                "text": chunks[idx]["text"],
                "similarity": float(similarities[idx]),
                "score": float(similarities[idx]),  # Score direct = similarité cosinus
                "rank": i + 1
            })
        
        return results
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Cosine Index '{index_name}' not found")

# Endpoints pour les modèles disponibles
@app.get("/models")
def get_models():
    return MODELS

# Endpoint pour liste des bases de données vectorielles supportées
@app.get("/vector_dbs")
def get_vector_dbs():
    supported_dbs = {
        "faiss": {
            "available": True,
            "description": "Facebook AI Similarity Search - Une bibliothèque optimisée pour la recherche de similarité à grande échelle"
        },
        "chroma": {
            "available": CHROMA_AVAILABLE,
            "description": "Une base de données vectorielle spécialement conçue pour les applications d'IA et les embeddings"
        },
        "lancedb": {
            "available": LANCEDB_AVAILABLE,
            "description": "Une base de données vectorielle orientée document optimisée pour les workloads vectoriels"
        },
        "cosine": {
            "available": True,
            "description": "Une méthode simple basée sur la similarité cosinus (sans base de données spécifique)"
        }
    }
    return supported_dbs

# Endpoint pour obtenir des embeddings
@app.post("/embeddings", response_model=EmbeddingResult)
def create_embeddings(request: EmbeddingRequest):
    # Vérifier que le modèle est disponible pour ce provider
    if request.model not in MODELS.get(request.provider, []):
        raise HTTPException(
            status_code=400, 
            detail=f"Model {request.model} not available for provider {request.provider}. Available models: {MODELS.get(request.provider)}"
        )
    
    try:
        # Normaliser les textes en liste
        texts = request.texts if isinstance(request.texts, list) else [request.texts]
        
        # Obtenir les embeddings selon le provider
        result = get_embeddings_by_provider(
            request.provider,
            request.model,
            texts,
            encoding_format=request.encoding_format,
            input_type=request.input_type
        )
        
        embeddings = result["embeddings"]
        total_tokens = result.get("total_tokens")
        
        # Calculer la dimension
        dimension = len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0
        
        return {
            "embeddings": embeddings,
            "model": request.model,
            "provider": request.provider,
            "dimension": dimension,
            "total_tokens": total_tokens
        }
    
    except Exception as e:
        logger.error(f"Error in embedding request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour créer un index
@app.post("/index", response_model=IndexInfo)
def create_index(request: CreateIndexRequest):
    # Vérifier que le modèle est disponible pour ce provider
    if request.model not in MODELS.get(request.provider, []):
        raise HTTPException(
            status_code=400, 
            detail=f"Model {request.model} not available for provider {request.provider}. Available models: {MODELS.get(request.provider)}"
        )
    
    # Vérifier la disponibilité de la base de données vectorielle
    if request.db_type == VectorDBType.CHROMA and not CHROMA_AVAILABLE:
        raise HTTPException(
            status_code=400, 
            detail="ChromaDB not installed. Please install with 'pip install chromadb'"
        )
    elif request.db_type == VectorDBType.LANCEDB and not LANCEDB_AVAILABLE:
        raise HTTPException(
            status_code=400, 
            detail="LanceDB not installed. Please install with 'pip install lancedb'"
        )
    
    try:
        # Obtenir les embeddings
        result = get_embeddings_by_provider(
            request.provider,
            request.model,
            request.texts,
            encoding_format=request.encoding_format,
            input_type=request.input_type
        )
        
        embeddings = result["embeddings"]
        
        # Paramètres supplémentaires spécifiques au provider
        extra_params = {}
        if request.provider == Provider.COHERE:
            extra_params["input_type"] = request.input_type
        elif request.provider in [Provider.OPENAI, Provider.MISTRAL, Provider.DEEPINFRA]:
            extra_params["encoding_format"] = request.encoding_format
        
        # Créer l'index selon la méthode choisie
        if request.db_type == VectorDBType.FAISS:
            _, metadata = create_faiss_index(
                request.provider, 
                request.model, 
                request.index_name, 
                request.texts, 
                embeddings, 
                **extra_params
            )
        elif request.db_type == VectorDBType.CHROMA:
            _, metadata = create_chroma_index(
                request.provider, 
                request.model, 
                request.index_name, 
                request.texts, 
                embeddings, 
                **extra_params
            )
        elif request.db_type == VectorDBType.LANCEDB:
            _, metadata = create_lancedb_index(
                request.provider, 
                request.model, 
                request.index_name, 
                request.texts, 
                embeddings, 
                **extra_params
            )
        else:  # VectorDBType.COSINE
            metadata = create_cosine_index(
                request.provider, 
                request.model, 
                request.index_name, 
                request.texts, 
                embeddings, 
                **extra_params
            )
        
        return {
            "provider": request.provider,
            "model": request.model,
            "index_name": request.index_name,
            "total_chunks": len(request.texts),
            "created_at": metadata["created_at"],
            "dimension": metadata["embedding_dim"],
            "db_type": request.db_type
        }
        
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour mettre à jour un index
@app.put("/index/{index_name}", response_model=IndexInfo)
def update_index(index_name: str, request: UpdateIndexRequest):
    # Vérifier que le modèle est disponible pour ce provider
    if request.model not in MODELS.get(request.provider, []):
        raise HTTPException(
            status_code=400, 
            detail=f"Model {request.model} not available for provider {request.provider}. Available models: {MODELS.get(request.provider)}"
        )
    
    # Vérifier la disponibilité de la base de données vectorielle
    if request.db_type == VectorDBType.CHROMA and not CHROMA_AVAILABLE:
        raise HTTPException(
            status_code=400, 
            detail="ChromaDB not installed. Please install with 'pip install chromadb'"
        )
    elif request.db_type == VectorDBType.LANCEDB and not LANCEDB_AVAILABLE:
        raise HTTPException(
            status_code=400, 
            detail="LanceDB not installed. Please install with 'pip install lancedb'"
        )
    
    try:
        # Obtenir les embeddings
        result = get_embeddings_by_provider(
            request.provider,
            request.model,
            request.texts,
            encoding_format="float",
            input_type="classification"
        )
        
        embeddings = result["embeddings"]
        
        # Mettre à jour l'index selon la méthode choisie
        if request.db_type == VectorDBType.FAISS:
            # Vérifier si l'index existe
            if not os.path.exists(f"{DATA_DIRECTORY}/faiss/{index_name}.faiss"):
                raise HTTPException(status_code=404, detail=f"FAISS index '{index_name}' not found")
                
            _, metadata = update_faiss_index(
                request.provider, 
                request.model, 
                index_name, 
                request.texts, 
                embeddings
            )
        elif request.db_type == VectorDBType.CHROMA:
            _, metadata = update_chroma_index(
                request.provider, 
                request.model, 
                index_name, 
                request.texts, 
                embeddings
            )
        elif request.db_type == VectorDBType.LANCEDB:
            _, metadata = update_lancedb_index(
                request.provider, 
                request.model, 
                index_name, 
                request.texts, 
                embeddings
            )
        else:  # VectorDBType.COSINE
            # Vérifier si l'index existe
            if not os.path.exists(f"{DATA_DIRECTORY}/json/{index_name}.json"):
                raise HTTPException(status_code=404, detail=f"Cosine index '{index_name}' not found")
                
            metadata = update_cosine_index(
                request.provider, 
                request.model, 
                index_name, 
                request.texts, 
                embeddings
            )
        
        return {
            "provider": request.provider,
            "model": request.model,
            "index_name": index_name,
            "total_chunks": metadata["total_chunks"],
            "created_at": metadata["created_at"],
            "updated_at": metadata.get("updated_at"),
            "dimension": metadata["embedding_dim"],
            "db_type": request.db_type
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error updating index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour rechercher
@app.post("/search", response_model=SearchResult)
def search(request: SearchRequest):
    # Vérifier que le modèle est disponible pour ce provider
    if request.model not in MODELS.get(request.provider, []):
        raise HTTPException(
            status_code=400, 
            detail=f"Model {request.model} not available for provider {request.provider}. Available models: {MODELS.get(request.provider)}"
        )
    
    # Vérifier la disponibilité de la base de données vectorielle
    if request.db_type == VectorDBType.CHROMA and not CHROMA_AVAILABLE:
        raise HTTPException(
            status_code=400, 
            detail="ChromaDB not installed. Please install with 'pip install chromadb'"
        )
    elif request.db_type == VectorDBType.LANCEDB and not LANCEDB_AVAILABLE:
        raise HTTPException(
            status_code=400, 
            detail="LanceDB not installed. Please install with 'pip install lancedb'"
        )
    
    try:
        # Obtenir l'embedding de la requête
        result = get_embeddings_by_provider(
            request.provider,
            request.model,
            request.query,
            encoding_format="float",
            input_type="classification"
        )
        
        query_embedding = result["embeddings"][0]
        
        # Effectuer la recherche selon la méthode choisie
        if request.db_type == VectorDBType.FAISS:
            search_results = search_faiss_index(request.index_name, query_embedding, request.top_k)
        elif request.db_type == VectorDBType.CHROMA:
            search_results = search_chroma_index(request.index_name, query_embedding, request.top_k)
        elif request.db_type == VectorDBType.LANCEDB:
            search_results = search_lancedb_index(request.index_name, query_embedding, request.top_k)
        else:  # VectorDBType.COSINE
            search_results = search_cosine_index(request.index_name, query_embedding, request.top_k)
        
        return {
            "index_name": request.index_name,
            "provider": request.provider,
            "model": request.model,
            "query": request.query,
            "db_type": request.db_type,
            "results": search_results
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour lister les index disponibles
@app.get("/indexes", response_model=List[IndexInfo])
def list_indexes():
    indexes = []
    
    try:
        # Parcourir les index FAISS
        if os.path.exists(f"{DATA_DIRECTORY}/faiss"):
            for filename in os.listdir(f"{DATA_DIRECTORY}/faiss"):
                if filename.endswith(".json"):
                    index_name = filename.split(".")[0]
                    
                    # Charger les métadonnées
                    with open(f"{DATA_DIRECTORY}/faiss/{filename}", 'r') as f:
                        metadata = json.load(f)
                    
                    # Créer l'objet IndexInfo
                    index_info = {
                        "provider": metadata.get("provider", "unknown"),
                        "model": metadata.get("model", "unknown"),
                        "index_name": index_name,
                        "total_chunks": metadata.get("total_chunks", 0),
                        "created_at": metadata.get("created_at", ""),
                        "updated_at": metadata.get("updated_at"),
                        "dimension": metadata.get("embedding_dim", 0),
                        "db_type": VectorDBType.FAISS
                    }
                    
                    indexes.append(index_info)
        
        # Parcourir les index ChromaDB
        if os.path.exists(f"{DATA_DIRECTORY}/chroma"):
            for filename in os.listdir(f"{DATA_DIRECTORY}/chroma"):
                if filename.endswith("_metadata.json"):
                    index_name = filename.replace("_metadata.json", "")
                    
                    # Charger les métadonnées
                    with open(f"{DATA_DIRECTORY}/chroma/{filename}", 'r') as f:
                        metadata = json.load(f)
                    
                    # Créer l'objet IndexInfo
                    index_info = {
                        "provider": metadata.get("provider", "unknown"),
                        "model": metadata.get("model", "unknown"),
                        "index_name": index_name,
                        "total_chunks": metadata.get("total_chunks", 0),
                        "created_at": metadata.get("created_at", ""),
                        "updated_at": metadata.get("updated_at"),
                        "dimension": metadata.get("embedding_dim", 0),
                        "db_type": VectorDBType.CHROMA
                    }
                    
                    indexes.append(index_info)
        
        # Parcourir les index LanceDB
        if os.path.exists(f"{DATA_DIRECTORY}/lancedb"):
            for filename in os.listdir(f"{DATA_DIRECTORY}/lancedb"):
                if filename.endswith("_metadata.json"):
                    index_name = filename.replace("_metadata.json", "")
                    
                    # Charger les métadonnées
                    with open(f"{DATA_DIRECTORY}/lancedb/{filename}", 'r') as f:
                        metadata = json.load(f)
                    
                    # Créer l'objet IndexInfo
                    index_info = {
                        "provider": metadata.get("provider", "unknown"),
                        "model": metadata.get("model", "unknown"),
                        "index_name": index_name,
                        "total_chunks": metadata.get("total_chunks", 0),
                        "created_at": metadata.get("created_at", ""),
                        "updated_at": metadata.get("updated_at"),
                        "dimension": metadata.get("embedding_dim", 0),
                        "db_type": VectorDBType.LANCEDB
                    }
                    
                    indexes.append(index_info)
        
        # Parcourir les index JSON (cosine)
        if os.path.exists(f"{DATA_DIRECTORY}/json"):
            for filename in os.listdir(f"{DATA_DIRECTORY}/json"):
                if filename.endswith(".json"):
                    index_name = filename.split(".")[0]
                    
                    # Charger les métadonnées
                    with open(f"{DATA_DIRECTORY}/json/{filename}", 'r') as f:
                        metadata = json.load(f)
                    
                    # Créer l'objet IndexInfo
                    index_info = {
                        "provider": metadata.get("provider", "unknown"),
                        "model": metadata.get("model", "unknown"),
                        "index_name": index_name,
                        "total_chunks": metadata.get("total_chunks", 0),
                        "created_at": metadata.get("created_at", ""),
                        "updated_at": metadata.get("updated_at"),
                        "dimension": metadata.get("embedding_dim", 0),
                        "db_type": VectorDBType.COSINE
                    }
                    
                    indexes.append(index_info)
        
        return indexes
        
    except Exception as e:
        logger.error(f"Error listing indexes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour supprimer un index
@app.delete("/index/{index_name}")
def delete_index(index_name: str, db_type: VectorDBType = Query(..., description="Type de base de données vectorielle utilisée")):
    try:
        # Vérifier si l'index existe selon la méthode
        if db_type == VectorDBType.FAISS:
            index_path = f"{DATA_DIRECTORY}/faiss/{index_name}.faiss"
            metadata_path = f"{DATA_DIRECTORY}/faiss/{index_name}.json"
            
            if not os.path.exists(index_path):
                raise HTTPException(status_code=404, detail=f"FAISS index '{index_name}' not found")
            
            # Supprimer les fichiers
            os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
        elif db_type == VectorDBType.CHROMA:
            if not CHROMA_AVAILABLE:
                raise HTTPException(status_code=400, detail="ChromaDB not installed")
                
            # Vérifier si les métadonnées existent
            metadata_path = f"{DATA_DIRECTORY}/chroma/{index_name}_metadata.json"
            if not os.path.exists(metadata_path):
                raise HTTPException(status_code=404, detail=f"ChromaDB index '{index_name}' not found")
            
            # Supprimer la collection
            client = chromadb.PersistentClient(path=f"{DATA_DIRECTORY}/chroma")
            try:
                client.delete_collection(name=index_name)
            except Exception as e:
                logger.error(f"Error deleting ChromaDB collection: {str(e)}")
            
            # Supprimer les métadonnées
            os.remove(metadata_path)
            
        elif db_type == VectorDBType.LANCEDB:
            if not LANCEDB_AVAILABLE:
                raise HTTPException(status_code=400, detail="LanceDB not installed")
                
            # Vérifier si les métadonnées existent
            metadata_path = f"{DATA_DIRECTORY}/lancedb/{index_name}_metadata.json"
            if not os.path.exists(metadata_path):
                raise HTTPException(status_code=404, detail=f"LanceDB index '{index_name}' not found")
            
            # Supprimer la table
            db = lancedb.connect(f"{DATA_DIRECTORY}/lancedb")
            try:
                db.drop_table(index_name)
            except Exception as e:
                logger.error(f"Error deleting LanceDB table: {str(e)}")
            
            # Supprimer les métadonnées
            os.remove(metadata_path)
            
        else:  # VectorDBType.COSINE
            index_path = f"{DATA_DIRECTORY}/json/{index_name}.json"
            
            if not os.path.exists(index_path):
                raise HTTPException(status_code=404, detail=f"Cosine index '{index_name}' not found")
            
            # Supprimer le fichier
            os.remove(index_path)
        
        return {"message": f"Index '{index_name}' ({db_type}) deleted successfully"}
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error deleting index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour obtenir les informations d'un index
@app.get("/index/{index_name}", response_model=IndexInfo)
def get_index_info(index_name: str, db_type: VectorDBType = Query(..., description="Type de base de données vectorielle utilisée")):
    try:
        # Charger les métadonnées selon la méthode
        if db_type == VectorDBType.FAISS:
            metadata_path = f"{DATA_DIRECTORY}/faiss/{index_name}.json"
            
            if not os.path.exists(metadata_path):
                raise HTTPException(status_code=404, detail=f"FAISS index '{index_name}' not found")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
        elif db_type == VectorDBType.CHROMA:
            metadata_path = f"{DATA_DIRECTORY}/chroma/{index_name}_metadata.json"
            
            if not os.path.exists(metadata_path):
                raise HTTPException(status_code=404, detail=f"ChromaDB index '{index_name}' not found")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
        elif db_type == VectorDBType.LANCEDB:
            metadata_path = f"{DATA_DIRECTORY}/lancedb/{index_name}_metadata.json"
            
            if not os.path.exists(metadata_path):
                raise HTTPException(status_code=404, detail=f"LanceDB index '{index_name}' not found")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
        else:  # VectorDBType.COSINE
            metadata_path = f"{DATA_DIRECTORY}/json/{index_name}.json"
            
            if not os.path.exists(metadata_path):
                raise HTTPException(status_code=404, detail=f"Cosine index '{index_name}' not found")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Créer l'objet IndexInfo
        index_info = {
            "provider": metadata.get("provider", "unknown"),
            "model": metadata.get("model", "unknown"),
            "index_name": index_name,
            "total_chunks": metadata.get("total_chunks", 0),
            "created_at": metadata.get("created_at", ""),
            "updated_at": metadata.get("updated_at"),
            "dimension": metadata.get("embedding_dim", 0),
            "db_type": db_type
        }
        
        return index_info
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error getting index info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Comparer deux embeddings et obtenir leur similarité
@app.post("/compare")
def compare_embeddings(
    texts: List[str] = Query(..., min_items=2, max_items=2, description="Deux textes à comparer"),
    provider: Provider = Query(..., description="Provider à utiliser"),
    model: str = Query(..., description="Modèle à utiliser"),
    encoding_format: str = Query("float", description="Format d'encodage (pour OpenAI, Mistral)"),
    input_type: str = Query("classification", description="Type d'entrée (pour Cohere)")
):
    # Vérifier que le modèle est disponible pour ce provider
    if model not in MODELS.get(provider, []):
        raise HTTPException(
            status_code=400, 
            detail=f"Model {model} not available for provider {provider}. Available models: {MODELS.get(provider)}"
        )
    
    try:
        # Obtenir les embeddings
        result = get_embeddings_by_provider(
            provider,
            model,
            texts,
            encoding_format=encoding_format,
            input_type=input_type
        )
        
        embeddings = result["embeddings"]
        
        # Calculer la similarité cosinus
        similarity = cosine_similarity(
            np.array(embeddings[0]).reshape(1, -1),
            np.array(embeddings[1]).reshape(1, -1)
        )[0][0]
        
        return {
            "text1": texts[0],
            "text2": texts[1],
            "similarity": float(similarity),
            "provider": provider,
            "model": model
        }
        
    except Exception as e:
        logger.error(f"Error comparing embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
