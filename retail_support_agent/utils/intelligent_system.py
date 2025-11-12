"""
Intelligent Retail System
Unified system combining Advanced RAG + Hybrid ML + LLM
"""

import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from utils.advanced_rag import AdvancedRAGSystem
from utils.hybrid_ml import HybridMLSystem


class IntelligentRetailSystem:
    """
    Unified Intelligent Retail System combining:
    1. Advanced RAG - Contextual product search
    2. Hybrid ML - Data analysis and insights
    3. LLM - Natural language interface and interpretation
    """

    def __init__(
        self,
        vectorstore_path: str = "./vectorstore/chroma_db",
        data_path: str = "./data/processed/processed_products.csv",
        use_openai_embeddings: bool = False,
        llm_model: str = "gpt-3.5-turbo"
    ):
        """Initialize Intelligent Retail System"""
        print("🌟" * 40)
        print("INTELLIGENT RETAIL SYSTEM")
        print("Advanced RAG + Hybrid ML + LLM Integration")
        print("🌟" * 40 + "\n")

        # Initialize Advanced RAG
        print("1️⃣ Initializing Advanced RAG System...")
        self.rag = AdvancedRAGSystem(
            vectorstore_path=vectorstore_path,
            use_openai_embeddings=use_openai_embeddings,
            llm_model=llm_model
        )

        # Initialize Hybrid ML
        print("\n2️⃣ Initializing Hybrid ML System...")
        self.ml = HybridMLSystem(
            data_path=data_path,
            llm_model=llm_model
        )

        # Initialize LLM for conversation
        print("\n3️⃣ Initializing LLM for conversation...")
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        print("   ✅ LLM ready!\n")

        # Cache for ML analysis
        self._ml_analysis_cache = None

        print("=" * 80)
        print("✅ INTELLIGENT RETAIL SYSTEM READY!")
        print("=" * 80 + "\n")

    def search_products(
        self,
        query: str,
        k: int = 3,
        use_advanced: bool = True
    ) -> str:
        """
        Search products using Advanced RAG
        Returns formatted natural language response
        """
        print(f"🔍 Searching for: '{query}'")

        try:
            # Use advanced RAG pipeline
            if use_advanced:
                results = self.rag.advanced_search(query, k=k)
            else:
                results = self.rag.simple_search(query, k=k)

            if not results:
                return "Sorry, no products found matching your query."

            # Format results with LLM
            return self._format_search_results(query, results)

        except Exception as e:
            print(f"❌ Search error: {e}")
            return f"Sorry, an error occurred during search: {str(e)}"

    def _format_search_results(self, query: str, results: List[Document]) -> str:
        """Format search results into natural language"""
        # Build context
        context = f"User searched for: '{query}'\n\nTop products found:\n\n"

        for i, doc in enumerate(results, 1):
            metadata = doc.metadata
            context += f"{i}. {metadata.get('product_name', 'Unknown Product')}\n"
            context += f"   Price: {metadata.get('price', 'N/A')}\n"
            context += f"   Rating: {metadata.get('rating', 'N/A')}/5 ({metadata.get('rating_count', 'N/A')} reviews)\n"
            context += f"   Category: {metadata.get('category', 'N/A')}\n"
            context += f"   Details: {doc.page_content[:300]}...\n\n"

        # LLM formats the response
        prompt = f"""{context}

Based on the search results above, provide a helpful response that:
1. Briefly introduces the top {len(results)} products found
2. Highlights key features that match the user's query
3. Mentions prices and ratings
4. Provides a recommendation

Keep it conversational and concise (3-4 sentences)."""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            # Fallback to simple formatting
            simple_response = f"Found {len(results)} products matching '{query}':\n\n"
            for i, doc in enumerate(results, 1):
                metadata = doc.metadata
                simple_response += f"{i}. **{metadata.get('product_name', 'Unknown')}**\n"
                simple_response += f"   💰 {metadata.get('price', 'N/A')} | "
                simple_response += f"⭐ {metadata.get('rating', 'N/A')}/5\n\n"
            return simple_response

    def get_recommendations(
        self,
        query: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get intelligent product recommendations
        Combines RAG search with ML insights
        """
        print(f"🎯 Getting recommendations for: '{query}'")

        try:
            # Search with Advanced RAG
            results = self.rag.advanced_search(query, k=5)

            if not results:
                return "Sorry, couldn't find products matching your criteria."

            # Extract product data
            products_info = []
            for doc in results:
                metadata = doc.metadata
                products_info.append({
                    'name': metadata.get('product_name', 'Unknown'),
                    'price': metadata.get('price', 'N/A'),
                    'rating': metadata.get('rating', 0),
                    'rating_count': metadata.get('rating_count', '0'),
                    'category': metadata.get('category', 'N/A')
                })

            # Get ML insights for context
            if preferences and preferences.get('include_analysis'):
                ml_insights = self._get_cached_ml_analysis()
            else:
                ml_insights = None

            # LLM generates recommendations
            return self._generate_recommendations(query, products_info, ml_insights)

        except Exception as e:
            print(f"❌ Recommendation error: {e}")
            return f"Sorry, couldn't generate recommendations: {str(e)}"

    def _generate_recommendations(
        self,
        query: str,
        products: List[Dict],
        ml_insights: Optional[Dict] = None
    ) -> str:
        """Generate natural language recommendations"""
        products_text = "\n".join([
            f"- {p['name']}: {p['price']}, {p['rating']}/5 rating ({p['rating_count']} reviews)"
            for p in products[:3]
        ])

        insights_text = ""
        if ml_insights:
            insights_text = f"\n\nMarket Insights:\n{ml_insights.get('price_segments', {}).get('insight', '')}"

        prompt = f"""Generate product recommendations based on:

User Query: "{query}"

Top Products:
{products_text}
{insights_text}

Provide:
1. Top 3 recommended products with brief reasons why
2. Value analysis (best price-to-quality ratio)
3. Any caveats or considerations

Format with markdown, use emojis, keep it under 250 words."""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            # Fallback
            fallback = f"**Recommendations for: {query}**\n\n"
            for i, p in enumerate(products[:3], 1):
                fallback += f"{i}. **{p['name']}**\n"
                fallback += f"   💰 {p['price']} | ⭐ {p['rating']}/5\n\n"
            return fallback

    def analyze_market(self, use_cache: bool = True) -> str:
        """
        Analyze market using Hybrid ML + LLM
        Returns comprehensive market insights
        """
        print("📊 Analyzing market with Hybrid ML + GenAI...")

        try:
            # Run ML analysis
            if use_cache and self._ml_analysis_cache:
                analysis = self._ml_analysis_cache
            else:
                analysis = self.ml.complete_analysis()
                self._ml_analysis_cache = analysis

            # Format comprehensive report
            report = "# 📊 Market Analysis Report\n\n"

            report += "## 🔹 Product Clustering\n"
            report += analysis['clustering']['insight'] + "\n\n"

            report += "## 🔹 Quality Analysis\n"
            report += analysis['classification']['insight'] + "\n\n"

            report += "## 🔹 Price Segments\n"
            report += analysis['price_segments']['insight'] + "\n\n"

            report += "## 🔹 Customer Sentiment\n"
            report += analysis['sentiment']['insight'] + "\n\n"

            # Add key metrics
            report += "## 📈 Key Metrics\n\n"
            sentiment_data = analysis['sentiment']['data']
            report += f"- **Total Products Analyzed:** {analysis['clustering']['data']['total_products']}\n"
            report += f"- **Average Rating:** {sentiment_data['avg_rating']:.2f}/5\n"
            report += f"- **Positive Sentiment:** {sentiment_data['positive_percentage']:.1f}%\n"
            report += f"- **High Quality Products:** {analysis['classification']['data']['quality_distribution']['high_quality']}\n"

            return report

        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return f"Sorry, market analysis failed: {str(e)}"

    def _get_cached_ml_analysis(self) -> Dict[str, Any]:
        """Get or create ML analysis cache"""
        if not self._ml_analysis_cache:
            self._ml_analysis_cache = self.ml.complete_analysis()
        return self._ml_analysis_cache

    def answer_question(self, question: str) -> str:
        """
        Answer user questions using RAG + ML insights
        Intelligent Q&A system
        """
        print(f"💬 Answering: '{question}'")

        question_lower = question.lower()

        # Route to appropriate handler
        if any(word in question_lower for word in ['search', 'find', 'show me', 'looking for']):
            return self.search_products(question, k=3)

        elif any(word in question_lower for word in ['recommend', 'suggest', 'best', 'should i buy']):
            return self.get_recommendations(question)

        elif any(word in question_lower for word in ['market', 'trend', 'analyze', 'insights', 'overview']):
            return self.analyze_market()

        elif any(word in question_lower for word in ['price', 'cheap', 'expensive', 'affordable', 'budget']):
            return self._answer_price_question(question)

        elif any(word in question_lower for word in ['quality', 'rating', 'review', 'good']):
            return self._answer_quality_question(question)

        else:
            # General question - use RAG + LLM
            return self._answer_general_question(question)

    def _answer_price_question(self, question: str) -> str:
        """Answer price-related questions"""
        # Get price insights
        analysis = self._get_cached_ml_analysis()
        price_data = analysis['price_segments']['data']

        context = f"""Price Segment Analysis:
Budget: {price_data['segments']['Budget']['count']} products, avg ₹{price_data['segments']['Budget']['avg_price']:.0f}
Mid-range: {price_data['segments']['Mid-range']['count']} products, avg ₹{price_data['segments']['Mid-range']['avg_price']:.0f}
Premium: {price_data['segments']['Premium']['count']} products, avg ₹{price_data['segments']['Premium']['avg_price']:.0f}

Insight: {analysis['price_segments']['insight']}"""

        prompt = f"""User question: "{question}"

{context}

Answer the user's question based on the price analysis. Be specific and helpful."""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except:
            return analysis['price_segments']['insight']

    def _answer_quality_question(self, question: str) -> str:
        """Answer quality-related questions"""
        analysis = self._get_cached_ml_analysis()

        context = f"""Quality Analysis:
High quality products: {analysis['classification']['data']['quality_distribution']['high_quality']}
Overall positive sentiment: {analysis['sentiment']['data']['positive_percentage']:.1f}%
Average rating: {analysis['sentiment']['data']['avg_rating']:.2f}/5

Insights: {analysis['classification']['insight']}"""

        prompt = f"""User question: "{question}"

{context}

Answer based on quality and sentiment analysis. Be helpful and specific."""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except:
            return analysis['classification']['insight']

    def _answer_general_question(self, question: str) -> str:
        """Answer general questions using RAG"""
        # Search for relevant products
        results = self.rag.advanced_search(question, k=3)

        if not results:
            return "I'm not sure how to answer that. Could you rephrase your question about products?"

        # Build context
        context = "Relevant product information:\n\n"
        for doc in results:
            context += f"Product: {doc.metadata.get('product_name', 'Unknown')}\n"
            context += f"{doc.page_content[:200]}...\n\n"

        prompt = f"""User question: "{question}"

{context}

Answer the user's question based on the product information above. Be helpful and concise."""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except:
            return "Based on the products available, I can help you find what you're looking for. Could you be more specific?"


def main():
    """Demo the complete intelligent system"""
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize system
    system = IntelligentRetailSystem(use_openai_embeddings=False)

    # Demo queries
    print("\n" + "=" * 80)
    print("DEMO 1: Product Search")
    print("=" * 80)
    result = system.search_products("affordable USB cable with fast charging")
    print(result)

    print("\n" + "=" * 80)
    print("DEMO 2: Recommendations")
    print("=" * 80)
    result = system.get_recommendations("charger cable under 300 rupees with good reviews")
    print(result)

    print("\n" + "=" * 80)
    print("DEMO 3: Market Analysis")
    print("=" * 80)
    result = system.analyze_market()
    print(result)

    print("\n" + "=" * 80)
    print("DEMO 4: Q&A")
    print("=" * 80)
    result = system.answer_question("What's the best value for money in this category?")
    print(result)


if __name__ == "__main__":
    main()
