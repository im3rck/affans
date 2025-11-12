"""
FREE Demo - No OpenAI API Key Required
Demonstrates Advanced RAG and ML features without LLM calls
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.advanced_rag import AdvancedRAGSystem
from utils.hybrid_ml import HybridMLSystem


def print_section(title: str, char: str = "="):
    """Print formatted section header"""
    print("\n" + char * 80)
    print(f"{title:^80}")
    print(char * 80 + "\n")


def demo_advanced_rag():
    """Demo Advanced RAG capabilities (no LLM needed for search)"""
    print_section("ADVANCED RAG SYSTEM - FREE DEMO", "=")

    print("Initializing Advanced RAG System (FREE local embeddings)...")
    rag = AdvancedRAGSystem(use_openai_embeddings=False)

    # Test 1: Simple Semantic Search
    print_section("Test 1: Semantic Search", "-")
    query = "USB cable"
    print(f"Query: '{query}'")
    results = rag.simple_search(query, k=3)
    print(f"\n📊 Top 3 Results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata.get('product_name', 'Unknown')}")
        print(f"   Price: {doc.metadata.get('price', 'N/A')}")
        print(f"   Rating: {doc.metadata.get('rating', 'N/A')}/5")
        print(f"   Category: {doc.metadata.get('category', 'N/A')[:60]}...")

    # Test 2: Hybrid Search
    print_section("Test 2: Hybrid Search (Semantic + BM25)", "-")
    query = "fast charging cable"
    print(f"Query: '{query}'")
    results = rag.hybrid_search(query, k=3)
    print(f"\n📊 Top 3 Results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata.get('product_name', 'Unknown')}")
        print(f"   Price: {doc.metadata.get('price', 'N/A')}")
        print(f"   Rating: {doc.metadata.get('rating', 'N/A')}/5")

    # Test 3: With Reranking (but no query rewriting to avoid LLM)
    print_section("Test 3: Search with Reranking", "-")
    query = "affordable charger"
    print(f"Query: '{query}'")

    # Get candidates with hybrid search
    candidates = rag.hybrid_search(query, k=10)

    # Rerank
    if rag.use_reranking:
        results = rag.rerank(query, candidates, top_k=3)
        print(f"\n🎯 Top 3 Results (after reranking):")
    else:
        results = candidates[:3]
        print(f"\n📊 Top 3 Results:")

    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata.get('product_name', 'Unknown')}")
        print(f"   Price: {doc.metadata.get('price', 'N/A')}")
        print(f"   Rating: {doc.metadata.get('rating', 'N/A')}/5")

    print("\n💡 Advanced RAG features demonstrated:")
    print("   ✅ Semantic search with local embeddings")
    print("   ✅ Hybrid search (semantic + BM25)")
    print("   ✅ Reranking with cross-encoder")
    print("   ⚠️  Query rewriting skipped (requires LLM)")


def demo_hybrid_ml():
    """Demo Hybrid ML capabilities (no LLM interpretation)"""
    print_section("HYBRID ML SYSTEM - FREE DEMO", "=")

    print("Initializing Hybrid ML System...")
    ml = HybridMLSystem()

    # Test 1: Product Clustering
    print_section("Test 1: Product Clustering", "-")
    clustering_result = ml.cluster_products(n_clusters=4)
    print("\n📊 Clustering Results:")
    for cluster in clustering_result['clusters']:
        print(f"\nCluster {cluster['cluster_id']}:")
        print(f"  Size: {cluster['size']} products")
        print(f"  Avg Price: ₹{cluster['avg_price']:.0f}")
        print(f"  Avg Rating: {cluster['avg_rating']:.2f}/5")
        print(f"  Price Range: ₹{cluster['price_range'][0]:.0f} - ₹{cluster['price_range'][1]:.0f}")

    # Test 2: Quality Classification
    print_section("Test 2: Quality Classification", "-")
    classification_result = ml.classify_quality()
    print("\n📊 Classification Results:")
    print(f"  Training accuracy: {classification_result['train_accuracy']:.1%}")
    print(f"  Test accuracy: {classification_result['test_accuracy']:.1%}")
    print(f"\n  Quality Distribution:")
    print(f"    High quality: {classification_result['quality_distribution']['high_quality']} products")
    print(f"    Lower quality: {classification_result['quality_distribution']['low_quality']} products")
    print(f"\n  Feature Importance:")
    for feature, importance in classification_result['feature_importance'].items():
        print(f"    {feature}: {importance:.3f}")

    # Test 3: Price Segmentation
    print_section("Test 3: Price Segmentation", "-")
    price_result = ml.analyze_price_segments()
    print("\n📊 Price Segments:")
    for segment_name, segment_info in price_result['segments'].items():
        print(f"\n{segment_name} Segment:")
        print(f"  Count: {segment_info['count']} products ({segment_info['percentage']:.1f}%)")
        print(f"  Price Range: ₹{segment_info['price_range'][0]:.0f} - ₹{segment_info['price_range'][1]:.0f}")
        print(f"  Avg Price: ₹{segment_info['avg_price']:.0f}")
        print(f"  Avg Rating: {segment_info['avg_rating']:.2f}/5")

    # Test 4: Sentiment Analysis
    print_section("Test 4: Sentiment Analysis", "-")
    sentiment_result = ml.sentiment_analysis()
    print("\n📊 Sentiment Distribution:")
    print(f"  Positive (4+ stars): {sentiment_result['positive']} ({sentiment_result['positive_percentage']:.1f}%)")
    print(f"  Neutral (3-4 stars): {sentiment_result['neutral']}")
    print(f"  Negative (<3 stars): {sentiment_result['negative']}")
    print(f"  Average Rating: {sentiment_result['avg_rating']:.2f}/5")

    print("\n💡 Hybrid ML features demonstrated:")
    print("   ✅ K-means clustering for product segmentation")
    print("   ✅ Random Forest quality classification")
    print("   ✅ Statistical price segmentation")
    print("   ✅ Sentiment analysis from ratings")
    print("   ⚠️  LLM interpretation skipped (requires API)")


def demo_comparison():
    """Demo showing comparison between techniques"""
    print_section("TECHNIQUE COMPARISON", "=")

    print("Initializing RAG system...")
    rag = AdvancedRAGSystem(use_openai_embeddings=False)

    query = "good cable under 300"

    # Simple search
    print_section("Baseline: Simple Semantic Search", "-")
    print(f"Query: '{query}'")
    results = rag.simple_search(query, k=3)
    print("\n📊 Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('product_name', 'Unknown')[:70]}")
        print(f"   Price: {doc.metadata.get('price', 'N/A')} | Rating: {doc.metadata.get('rating', 'N/A')}/5")

    # Hybrid search
    print_section("Enhanced: Hybrid Search (Semantic + BM25)", "-")
    print(f"Query: '{query}'")
    results = rag.hybrid_search(query, k=3)
    print("\n📊 Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('product_name', 'Unknown')[:70]}")
        print(f"   Price: {doc.metadata.get('price', 'N/A')} | Rating: {doc.metadata.get('rating', 'N/A')}/5")

    # With reranking
    if rag.use_reranking:
        print_section("Advanced: Hybrid Search + Reranking", "-")
        print(f"Query: '{query}'")
        candidates = rag.hybrid_search(query, k=10)
        results = rag.rerank(query, candidates, top_k=3)
        print("\n📊 Results:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.metadata.get('product_name', 'Unknown')[:70]}")
            print(f"   Price: {doc.metadata.get('price', 'N/A')} | Rating: {doc.metadata.get('rating', 'N/A')}/5")

    print("\n💡 Notice how each technique improves the results!")
    print("   • Semantic search understands meaning")
    print("   • BM25 adds keyword precision")
    print("   • Reranking refines the final ranking")


def main():
    """Run FREE demo (no API key required)"""
    load_dotenv()

    print("=" * 80)
    print("INTELLIGENT RETAIL SYSTEM - FREE DEMO")
    print("No OpenAI API Key Required!")
    print("=" * 80)
    print()
    print("This demo shows all features that work WITHOUT OpenAI API:")
    print("  • Advanced RAG (search with local embeddings)")
    print("  • Hybrid ML (clustering, classification, analysis)")
    print("  • Technique comparisons")
    print()
    print("Features requiring OpenAI API (skipped in this demo):")
    print("  • Query rewriting")
    print("  • LLM interpretation of ML results")
    print("  • Natural language Q&A")
    print()

    try:
        # Run demos
        print("\n🚀 Starting FREE demos...\n")

        # Part 1: Advanced RAG
        demo_advanced_rag()

        # Part 2: Hybrid ML
        demo_hybrid_ml()

        # Part 3: Comparison
        demo_comparison()

        # Summary
        print_section("FREE DEMO COMPLETE!", "=")
        print("✅ Demonstrated (without OpenAI API):")
        print("   1. Advanced RAG (Semantic + Hybrid Search + Reranking)")
        print("   2. Hybrid ML (Clustering, Classification, Price Analysis, Sentiment)")
        print("   3. Technique Comparisons")
        print()
        print("🎉 All FREE features working!")
        print()
        print("💡 To enable LLM features:")
        print("   1. Get OpenAI API key from https://platform.openai.com/")
        print("   2. Add to .env file: OPENAI_API_KEY=your-key-here")
        print("   3. Add $5 credit to your account")
        print("   4. Run: python demo.py")
        print()
        print("💡 Or use the web interface:")
        print("   streamlit run app_new.py")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure vectorstore exists: python setup_system.py")
        print("2. Check data is processed")
        print("3. Install dependencies: pip install -r requirements.txt")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
