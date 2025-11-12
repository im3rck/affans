"""
Advanced RAG System with Query Rewriting, Hybrid Search, and Reranking
Implements state-of-the-art retrieval techniques for improved accuracy
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder


class AdvancedRAGSystem:
    """
    Advanced RAG System with multiple retrieval enhancement techniques:
    1. Query Rewriting - Expands and clarifies user queries
    2. Hybrid Search - Combines semantic (embeddings) + keyword (BM25) search
    3. Reranking - Uses cross-encoder to reorder results by relevance
    4. Multi-Query - Generates query variations for comprehensive retrieval
    """

    def __init__(
        self,
        vectorstore_path: str = "./vectorstore/chroma_db",
        use_openai_embeddings: bool = False,
        llm_model: str = "gpt-3.5-turbo",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """Initialize Advanced RAG System"""
        print("🚀 Initializing Advanced RAG System...")

        # Initialize embeddings
        if use_openai_embeddings:
            print("   Using OpenAI embeddings...")
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            print("   Using FREE local HuggingFace embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        # Initialize LLM for query rewriting
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)

        # Load vectorstore
        print(f"   Loading vectorstore from {vectorstore_path}...")
        self.vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=self.embeddings
        )

        # Initialize BM25 for keyword search
        print("   Initializing BM25 keyword search...")
        self.documents = []
        self.bm25 = None
        self._initialize_bm25()

        # Initialize cross-encoder for reranking
        print(f"   Loading reranker model: {reranker_model}...")
        try:
            self.reranker = CrossEncoder(reranker_model)
            self.use_reranking = True
            print("   ✅ Reranker loaded successfully")
        except Exception as e:
            print(f"   ⚠️ Could not load reranker: {e}")
            print("   Continuing without reranking...")
            self.use_reranking = False

        print("✅ Advanced RAG System initialized!\n")

    def _initialize_bm25(self):
        """Initialize BM25 with all documents from vectorstore"""
        try:
            # Get all documents
            results = self.vectorstore.get()
            if results and 'documents' in results:
                self.documents = [
                    Document(
                        page_content=content,
                        metadata=metadata
                    )
                    for content, metadata in zip(
                        results['documents'],
                        results['metadatas']
                    )
                ]

                # Tokenize documents for BM25
                tokenized_docs = [doc.page_content.lower().split() for doc in self.documents]
                self.bm25 = BM25Okapi(tokenized_docs)
                print(f"   BM25 initialized with {len(self.documents)} documents")
        except Exception as e:
            print(f"   Warning: Could not initialize BM25: {e}")
            self.bm25 = None

    def rewrite_query(self, query: str) -> List[str]:
        """
        Query Rewriting: Expands user query into multiple variations
        Handles abbreviations, clarifies intent, and creates alternatives
        """
        try:
            prompt = f"""Given this user query about products: "{query}"

Rewrite it into 3 different variations that:
1. Expand abbreviations (USB → Universal Serial Bus)
2. Add synonyms (cheap → affordable, budget-friendly, inexpensive)
3. Clarify intent (good cable → high-quality durable cable)

Return ONLY the 3 rewritten queries, one per line, no numbering or explanations.

Query 1:
Query 2:
Query 3:"""

            response = self.llm.invoke(prompt)

            # Parse response
            rewritten = response.content.strip().split('\n')
            queries = [q.strip() for q in rewritten if q.strip() and not q.startswith('Query')]

            # Add original query
            all_queries = [query] + queries[:3]

            print(f"📝 Query Rewriting:")
            print(f"   Original: {query}")
            for i, q in enumerate(queries[:3], 1):
                print(f"   Variant {i}: {q}")

            return all_queries

        except Exception as e:
            print(f"⚠️ Query rewriting failed: {e}")
            return [query]

    def semantic_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Semantic search using embeddings"""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # Convert to (doc, score) format with normalized scores
        return [(doc, 1.0 / (1.0 + score)) for doc, score in results]

    def keyword_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Keyword search using BM25"""
        if not self.bm25:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_idx = np.argsort(scores)[::-1][:k]

        # Return documents with scores
        results = []
        for idx in top_k_idx:
            if idx < len(self.documents):
                # Normalize BM25 score to 0-1 range
                normalized_score = scores[idx] / (scores[idx] + 1.0)
                results.append((self.documents[idx], normalized_score))

        return results

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Document]:
        """
        Hybrid Search: Combines semantic and keyword search
        Uses weighted scoring to balance both approaches
        """
        print(f"🔍 Hybrid Search for: '{query}'")

        # Semantic search
        semantic_results = self.semantic_search(query, k=k*2)
        print(f"   Semantic: {len(semantic_results)} results")

        # Keyword search
        keyword_results = self.keyword_search(query, k=k*2)
        print(f"   Keyword (BM25): {len(keyword_results)} results")

        # Combine scores
        doc_scores = {}

        # Add semantic scores
        for doc, score in semantic_results:
            doc_id = doc.page_content[:100]  # Use content prefix as ID
            doc_scores[doc_id] = {
                'doc': doc,
                'semantic': score * semantic_weight,
                'keyword': 0
            }

        # Add keyword scores
        for doc, score in keyword_results:
            doc_id = doc.page_content[:100]
            if doc_id in doc_scores:
                doc_scores[doc_id]['keyword'] = score * keyword_weight
            else:
                doc_scores[doc_id] = {
                    'doc': doc,
                    'semantic': 0,
                    'keyword': score * keyword_weight
                }

        # Calculate final scores
        for doc_id in doc_scores:
            doc_scores[doc_id]['final'] = (
                doc_scores[doc_id]['semantic'] +
                doc_scores[doc_id]['keyword']
            )

        # Sort by final score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['final'],
            reverse=True
        )

        print(f"   Combined: Top {min(k, len(sorted_docs))} results")

        return [item['doc'] for item in sorted_docs[:k]]

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Reranking: Uses cross-encoder to reorder results by relevance
        More accurate but slower than initial retrieval
        """
        if not self.use_reranking or not documents:
            return documents[:top_k]

        print(f"🎯 Reranking {len(documents)} results...")

        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get relevance scores
        scores = self.reranker.predict(pairs)

        # Sort by score
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        print(f"   Top score: {doc_score_pairs[0][1]:.4f}")
        print(f"   Returning top {top_k} results")

        return [doc for doc, _ in doc_score_pairs[:top_k]]

    def multi_query_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Multi-Query: Generates query variations and aggregates results
        Provides comprehensive coverage of the information need
        """
        print(f"\n🔄 Multi-Query Search")

        # Generate query variations
        queries = self.rewrite_query(query)

        # Search with each query
        all_results = []
        seen_contents = set()

        for q in queries:
            results = self.hybrid_search(q, k=k*2)

            # Deduplicate
            for doc in results:
                content_hash = doc.page_content[:200]
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_results.append(doc)

        print(f"   Total unique results: {len(all_results)}")

        return all_results[:k*2]  # Return more for reranking

    def advanced_search(
        self,
        query: str,
        k: int = 3,
        use_query_rewriting: bool = True,
        use_reranking: bool = True
    ) -> List[Document]:
        """
        Complete Advanced RAG pipeline:
        1. Query Rewriting (optional)
        2. Multi-Query or Hybrid Search
        3. Reranking (optional)

        This is the main method to use for best results
        """
        print("=" * 80)
        print("🚀 ADVANCED RAG PIPELINE")
        print("=" * 80)

        try:
            # Step 1: Multi-query search (includes query rewriting and hybrid search)
            if use_query_rewriting:
                candidates = self.multi_query_search(query, k=k)
            else:
                candidates = self.hybrid_search(query, k=k*3)

            # Step 2: Reranking
            if use_reranking and self.use_reranking:
                final_results = self.rerank(query, candidates, top_k=k)
            else:
                final_results = candidates[:k]

            print("=" * 80)
            print(f"✅ Returned {len(final_results)} results")
            print("=" * 80 + "\n")

            return final_results

        except Exception as e:
            print(f"❌ Error in advanced search: {e}")
            # Fallback to simple search
            print("   Falling back to simple semantic search...")
            return self.vectorstore.similarity_search(query, k=k)

    def simple_search(self, query: str, k: int = 3) -> List[Document]:
        """Simple semantic search (fallback)"""
        return self.vectorstore.similarity_search(query, k=k)


def main():
    """Demo the Advanced RAG system"""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 80)
    print("ADVANCED RAG SYSTEM DEMO")
    print("=" * 80 + "\n")

    # Initialize
    rag = AdvancedRAGSystem(use_openai_embeddings=False)

    # Test query
    query = "affordable USB cable with good reviews"

    print("\n" + "=" * 80)
    print("TEST 1: Complete Advanced RAG Pipeline")
    print("=" * 80)
    results = rag.advanced_search(query, k=3)

    print("\n📊 RESULTS:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata.get('product_name', 'Unknown')}")
        print(f"   Price: {doc.metadata.get('price', 'N/A')}")
        print(f"   Rating: {doc.metadata.get('rating', 'N/A')}/5")
        print(f"   Category: {doc.metadata.get('category', 'N/A')[:60]}...")

    print("\n" + "=" * 80)
    print("TEST 2: Hybrid Search Only")
    print("=" * 80)
    results = rag.hybrid_search(query, k=3)

    print("\n📊 RESULTS:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata.get('product_name', 'Unknown')}")
        print(f"   Price: {doc.metadata.get('price', 'N/A')}")
        print(f"   Rating: {doc.metadata.get('rating', 'N/A')}/5")


if __name__ == "__main__":
    main()
