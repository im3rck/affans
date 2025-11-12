"""
RAG (Retrieval Augmented Generation) System
Handles vector database creation, embeddings, and retrieval
"""

import os
import json
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


class RAGSystem:
    """RAG system for product information retrieval"""

    def __init__(
        self,
        collection_name: str = "amazon_products",
        persist_directory: str = "./vectorstore/chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_openai: bool = False  # Default to FREE local embeddings
    ):
        """
        Initialize RAG system

        Args:
            collection_name: Name for the vector store collection
            persist_directory: Directory to persist vector store
            embedding_model: Model name for embeddings
            use_openai: If True, use OpenAI embeddings (requires API key).
                       If False, use free local HuggingFace embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.use_openai = use_openai

        # Initialize embeddings
        if use_openai:
            try:
                print("Initializing OpenAI embeddings...")
                self.embeddings = OpenAIEmbeddings(model=embedding_model)
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI embeddings: {e}")
                print("Falling back to free local embeddings...")
                self.use_openai = False
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
        else:
            print("Using free local HuggingFace embeddings (no API key required)...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        self.vectorstore = None
        self.documents = []

    def load_documents(self, json_path: str) -> List[Document]:
        """Load documents from JSON file"""
        print(f"Loading documents from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)

        documents = []
        for item in data:
            doc = Document(
                page_content=item['text'],
                metadata=item['metadata']
            )
            documents.append(doc)

        self.documents = documents
        print(f"Loaded {len(documents)} documents")
        return documents

    def chunk_documents(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """Split documents into chunks for better retrieval"""
        print(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunked_docs = text_splitter.split_documents(self.documents)
        print(f"Created {len(chunked_docs)} chunks from {len(self.documents)} documents")
        return chunked_docs

    def create_vectorstore(self, documents: Optional[List[Document]] = None):
        """Create or load vector store"""
        if documents is None:
            documents = self.documents

        print(f"Creating vector store with {len(documents)} documents...")

        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

        # Create Chroma vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )

        print(f"Vector store created and persisted to {self.persist_directory}")
        return self.vectorstore

    def load_vectorstore(self):
        """Load existing vector store"""
        print(f"Loading vector store from {self.persist_directory}...")

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        print("Vector store loaded successfully")
        return self.vectorstore

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """Perform similarity search"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore() or load_vectorstore() first.")

        print(f"Searching for: '{query}' (top {k} results)")

        if filter_dict:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3
    ) -> List[tuple]:
        """Perform similarity search with relevance scores"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """Get retriever for use with LangChain"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")

        if search_kwargs is None:
            search_kwargs = {"k": 3}

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def retrieve_context(
        self,
        query: str,
        k: int = 3,
        include_metadata: bool = True
    ) -> str:
        """Retrieve and format context for LLM"""
        results = self.similarity_search(query, k=k)

        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"[Document {i}]")
            context_parts.append(doc.page_content)

            if include_metadata and doc.metadata:
                context_parts.append(f"Metadata: {json.dumps(doc.metadata, indent=2)}")

            context_parts.append("")  # Add blank line

        return "\n".join(context_parts)

    def get_product_by_name(self, product_name: str, k: int = 1) -> List[Document]:
        """Retrieve specific product by name"""
        query = f"Product: {product_name}"
        return self.similarity_search(query, k=k)

    def get_products_by_category(self, category: str, k: int = 5) -> List[Document]:
        """Retrieve products by category"""
        filter_dict = {"category": category}
        results = self.vectorstore.similarity_search(
            category,
            k=k,
            filter=filter_dict
        )
        return results

    def get_statistics(self) -> Dict:
        """Get vector store statistics"""
        if self.vectorstore is None:
            return {"error": "Vector store not initialized"}

        collection = self.vectorstore._collection
        count = collection.count()

        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model_name
        }


def setup_rag_system(documents_path: str, persist_directory: str, use_openai: bool = False) -> RAGSystem:
    """
    Complete RAG system setup pipeline

    Args:
        documents_path: Path to processed documents JSON
        persist_directory: Directory to store vector database
        use_openai: If True, use OpenAI embeddings (requires API key).
                   If False, use free local HuggingFace embeddings (default)
    """
    print("="*60)
    print("Setting up RAG System")
    print("="*60)

    if not use_openai:
        print("💡 Using FREE local embeddings (no API key required)")
        print("   This may take a few minutes to download the model first time...")

    # Initialize RAG system with chosen embedding type
    rag = RAGSystem(persist_directory=persist_directory, use_openai=use_openai)

    # Load documents
    documents = rag.load_documents(documents_path)

    # Chunk documents for better retrieval
    chunked_docs = rag.chunk_documents(chunk_size=1000, chunk_overlap=200)

    # Create vector store
    rag.create_vectorstore(chunked_docs)

    # Display statistics
    stats = rag.get_statistics()
    print("\n" + "="*60)
    print("RAG System Setup Complete!")
    print("="*60)
    print(f"Total documents: {stats['total_documents']}")
    print(f"Collection: {stats['collection_name']}")
    print(f"Embedding model: {stats['embedding_model']}")

    return rag


def main():
    """Test RAG system"""
    # Setup
    rag = setup_rag_system(
        documents_path="../data/processed/rag_documents.json",
        persist_directory="../vectorstore/chroma_db"
    )

    # Test queries
    test_queries = [
        "What are the best USB cables?",
        "Show me products with good ratings",
        "Tell me about charging cables"
    ]

    print("\n" + "="*60)
    print("Testing RAG System")
    print("="*60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = rag.similarity_search(query, k=2)
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Product: {doc.metadata.get('product_name', 'N/A')}")
            print(f"Rating: {doc.metadata.get('rating', 'N/A')}")
            print(f"Content preview: {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
