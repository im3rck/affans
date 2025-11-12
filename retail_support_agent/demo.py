"""
Complete Demo of Intelligent Retail System
Advanced RAG + Hybrid ML + LLM Integration
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.intelligent_system import IntelligentRetailSystem
from utils.advanced_rag import AdvancedRAGSystem
from utils.hybrid_ml import HybridMLSystem


def print_section(title: str, char: str = "="):
    """Print formatted section header"""
    print("\n" + char * 80)
    print(f"{title:^80}")
    print(char * 80 + "\n")


def demo_advanced_rag():
    """Demo Advanced RAG capabilities"""
    print_section("PART 1: ADVANCED RAG SYSTEM", "=")

    print("Initializing Advanced RAG System...")
    rag = AdvancedRAGSystem(use_openai_embeddings=False)

    # Test 1: Query Rewriting
    print_section("Test 1: Query Rewriting", "-")
    query = "cheap cable"
    print(f"Original Query: '{query}'")
    rewritten = rag.rewrite_query(query)
    print(f"\n✅ Generated {len(rewritten)} query variations!")

    # Test 2: Hybrid Search
    print_section("Test 2: Hybrid Search (Semantic + BM25)", "-")
    query = "USB cable fast charging"
    results = rag.hybrid_search(query, k=3)
    print(f"\n📊 Top 3 Results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata.get('product_name', 'Unknown')}")
        print(f"   Price: {doc.metadata.get('price', 'N/A')}")
        print(f"   Rating: {doc.metadata.get('rating', 'N/A')}/5")

    # Test 3: Complete Advanced RAG Pipeline
    print_section("Test 3: Complete Advanced RAG Pipeline", "-")
    query = "affordable charger with good reviews"
    results = rag.advanced_search(query, k=3, use_query_rewriting=True, use_reranking=True)
    print(f"\n🎯 Final Results (after rewriting, hybrid search, and reranking):")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata.get('product_name', 'Unknown')}")
        print(f"   Price: {doc.metadata.get('price', 'N/A')}")
        print(f"   Rating: {doc.metadata.get('rating', 'N/A')}/5")
        print(f"   Category: {doc.metadata.get('category', 'N/A')[:60]}...")


def demo_hybrid_ml():
    """Demo Hybrid ML + GenAI capabilities"""
    print_section("PART 2: HYBRID ML + GENAI SYSTEM", "=")

    print("Initializing Hybrid ML System...")
    ml = HybridMLSystem()

    # Test 1: Product Clustering
    print_section("Test 1: Product Clustering", "-")
    clustering_result = ml.cluster_products(n_clusters=4)
    print("\n✅ Clustering complete! Generated insights:")
    insight = ml.interpret_with_llm('clustering', clustering_result)
    print(f"\n💡 LLM Insight:\n{insight}")

    # Test 2: Quality Classification
    print_section("Test 2: Quality Classification", "-")
    classification_result = ml.classify_quality()
    print("\n✅ Classification complete! Getting LLM interpretation:")
    insight = ml.interpret_with_llm('classification', classification_result)
    print(f"\n💡 LLM Insight:\n{insight}")

    # Test 3: Price Segmentation
    print_section("Test 3: Price Segmentation Analysis", "-")
    price_result = ml.analyze_price_segments()
    print("\n✅ Price analysis complete! Getting LLM interpretation:")
    insight = ml.interpret_with_llm('price_segments', price_result)
    print(f"\n💡 LLM Insight:\n{insight}")

    # Test 4: Sentiment Analysis
    print_section("Test 4: Sentiment Analysis", "-")
    sentiment_result = ml.sentiment_analysis()
    print("\n✅ Sentiment analysis complete! Getting LLM interpretation:")
    insight = ml.interpret_with_llm('sentiment', sentiment_result)
    print(f"\n💡 LLM Insight:\n{insight}")


def demo_intelligent_system():
    """Demo Complete Intelligent System"""
    print_section("PART 3: INTELLIGENT RETAIL SYSTEM (UNIFIED)", "=")

    print("Initializing Intelligent Retail System...")
    system = IntelligentRetailSystem(use_openai_embeddings=False)

    # Test 1: Product Search
    print_section("Test 1: Intelligent Product Search", "-")
    query = "affordable USB cable with fast charging and good reviews"
    print(f"Query: '{query}'\n")
    result = system.search_products(query, k=3, use_advanced=True)
    print("🔍 Search Result:")
    print(result)

    # Test 2: Recommendations
    print_section("Test 2: Smart Recommendations", "-")
    query = "I need a charger cable under 300 rupees with warranty"
    print(f"Query: '{query}'\n")
    result = system.get_recommendations(query, preferences={'include_analysis': True})
    print("🎯 Recommendations:")
    print(result)

    # Test 3: Market Analysis
    print_section("Test 3: Market Analysis", "-")
    print("Running complete market analysis with Hybrid ML + GenAI...\n")
    result = system.analyze_market(use_cache=True)
    print(result)

    # Test 4: Q&A
    print_section("Test 4: Intelligent Q&A", "-")
    questions = [
        "What are the best value products?",
        "Which price segment offers the best quality?",
        "Show me products with high ratings"
    ]

    for q in questions:
        print(f"\n❓ Question: {q}")
        answer = system.answer_question(q)
        print(f"💬 Answer: {answer}\n")
        print("-" * 80)


def demo_comparison():
    """Demo showing comparison between techniques"""
    print_section("PART 4: TECHNIQUE COMPARISON", "=")

    print("Initializing RAG system...")
    rag = AdvancedRAGSystem(use_openai_embeddings=False)

    query = "cheap but good cable"

    # Simple search
    print_section("Baseline: Simple Semantic Search", "-")
    print(f"Query: '{query}'")
    results = rag.simple_search(query, k=3)
    print("\n📊 Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('product_name', 'Unknown')[:60]}")

    # Hybrid search
    print_section("Enhanced: Hybrid Search (Semantic + BM25)", "-")
    print(f"Query: '{query}'")
    results = rag.hybrid_search(query, k=3)
    print("\n📊 Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('product_name', 'Unknown')[:60]}")

    # Advanced RAG
    print_section("Advanced: Complete RAG Pipeline", "-")
    print(f"Query: '{query}'")
    results = rag.advanced_search(query, k=3)
    print("\n📊 Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('product_name', 'Unknown')[:60]}")

    print("\n💡 Notice how each technique improves the results!")


def main():
    """Run complete demo"""
    load_dotenv()

    print("=" * 80)
    print("INTELLIGENT RETAIL SYSTEM - COMPLETE DEMO")
    print("Advanced RAG + Hybrid ML + LLM Integration")
    print("=" * 80)

    try:
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("\n⚠️ WARNING: No OpenAI API key found!")
            print("Some features (LLM interpretation, recommendations) will fail.")
            print("Set OPENAI_API_KEY in .env file to enable all features.\n")
            proceed = input("Continue anyway? (y/n): ")
            if proceed.lower() != 'y':
                print("Demo cancelled.")
                return

        # Run demos
        print("\n🚀 Starting demos...\n")

        # Part 1: Advanced RAG
        demo_advanced_rag()

        # Part 2: Hybrid ML
        demo_hybrid_ml()

        # Part 3: Intelligent System
        demo_intelligent_system()

        # Part 4: Comparison
        demo_comparison()

        # Summary
        print_section("DEMO COMPLETE!", "=")
        print("✅ Demonstrated:")
        print("   1. Advanced RAG (Query Rewriting, Hybrid Search, Reranking)")
        print("   2. Hybrid ML + GenAI (Clustering, Classification, LLM Interpretation)")
        print("   3. Intelligent System (Unified Search, Recommendations, Analysis, Q&A)")
        print("   4. Technique Comparison (Simple → Hybrid → Advanced)")
        print("\n🎉 All features working!")
        print("\n💡 Next steps:")
        print("   - Run Streamlit app: streamlit run app_new.py")
        print("   - View documentation: README.md")
        print("   - Check architecture: PROBLEM_SOLUTION.md")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure vectorstore exists: python utils/rag_system.py")
        print("2. Check data is processed: python utils/data_preprocessor.py")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Set OpenAI API key in .env file")


if __name__ == "__main__":
    main()
