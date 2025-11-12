"""
Simple RAG-based recommendation system (No OpenAI API required for demo)
Uses only vector search and templated responses
"""

from utils.rag_system import RAGSystem


class SimpleRecommendationSystem:
    """Simple recommendation system using only RAG (no LLM needed)"""

    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system

    def search_products(self, query: str, k: int = 5) -> str:
        """Search products and format results"""
        try:
            results = self.rag_system.similarity_search(query, k=k)

            if not results:
                return "Sorry, no products found matching your query."

            response = f"🔍 **Found {len(results)} products matching your search:**\n\n"

            for i, doc in enumerate(results, 1):
                metadata = doc.metadata

                # Extract key info
                name = metadata.get('product_name', 'Unknown Product')
                price = metadata.get('price', 'N/A')
                rating = metadata.get('rating', 'N/A')
                category = metadata.get('category', 'N/A')

                # Create product card
                response += f"**{i}. {name}**\n"
                response += f"   💰 Price: {price}\n"
                response += f"   ⭐ Rating: {rating}/5\n"
                response += f"   📦 Category: {category[:50]}...\n"

                # Extract features from content
                content = doc.page_content
                if "Product Details:" in content:
                    details = content.split("Product Details:")[1].split("Customer Reviews:")[0]
                    features = details.strip().split("|")[:3]  # Top 3 features
                    if features:
                        response += f"   ✨ Features: {', '.join([f.strip()[:80] for f in features if f.strip()])}\n"

                response += "\n"

            return response

        except Exception as e:
            return f"Error searching products: {str(e)}"

    def recommend_products(self, query: str) -> str:
        """Get recommendations with analysis"""
        try:
            results = self.rag_system.similarity_search(query, k=3)

            if not results:
                return "Sorry, no products found matching your criteria."

            response = f"🎯 **Recommendations based on: '{query}'**\n\n"

            # Analyze results
            prices = []
            ratings = []

            for i, doc in enumerate(results, 1):
                metadata = doc.metadata
                name = metadata.get('product_name', 'Unknown')
                price_str = metadata.get('price', '₹0')
                rating = float(metadata.get('rating', 0))

                # Extract price number
                price_num = 0
                try:
                    price_num = int(''.join(filter(str.isdigit, price_str.split(',')[0])))
                except:
                    pass

                if price_num > 0:
                    prices.append(price_num)
                if rating > 0:
                    ratings.append(rating)

                # Product recommendation
                response += f"**{i}. {name}**\n"
                response += f"   💰 {price_str}"

                # Value indicator
                if price_num > 0:
                    if price_num < 200:
                        response += " 🔥 (Great Value!)"
                    elif price_num > 400:
                        response += " 💎 (Premium)"

                response += f"\n   ⭐ {rating}/5"

                # Rating indicator
                if rating >= 4.2:
                    response += " 🌟 (Highly Rated!)"
                elif rating >= 4.0:
                    response += " 👍 (Good Reviews)"

                # Why recommend
                content = doc.page_content.lower()
                reasons = []

                if "fast charging" in content or "quick charge" in content:
                    reasons.append("Fast charging support")
                if "durable" in content or "durability" in content:
                    reasons.append("Durable design")
                if "warranty" in content:
                    reasons.append("Warranty included")
                if rating >= 4.0:
                    reasons.append(f"Excellent {rating}/5 rating")

                if reasons:
                    response += f"\n   ✅ Why: {', '.join(reasons[:3])}\n"

                response += "\n"

            # Overall analysis
            if prices and ratings:
                avg_price = sum(prices) / len(prices)
                avg_rating = sum(ratings) / len(ratings)

                response += "📊 **Analysis:**\n"
                response += f"   • Average Price: ₹{int(avg_price)}\n"
                response += f"   • Average Rating: {avg_rating:.1f}/5\n"

                # Best value
                best_idx = 0
                best_score = 0
                for i in range(len(results)):
                    if i < len(ratings) and i < len(prices):
                        # Score = rating / normalized_price
                        score = ratings[i] / (prices[i] / 100)
                        if score > best_score:
                            best_score = score
                            best_idx = i

                response += f"\n💡 **Best Value:** Option #{best_idx + 1}\n"

            return response

        except Exception as e:
            return f"Error generating recommendations: {str(e)}"

    def analyze_reviews(self, product_query: str) -> str:
        """Analyze product reviews"""
        try:
            results = self.rag_system.similarity_search(product_query, k=1)

            if not results:
                return "Product not found."

            doc = results[0]
            metadata = doc.metadata
            name = metadata.get('product_name', 'Unknown')
            rating = metadata.get('rating', 'N/A')
            rating_count = metadata.get('rating_count', 'N/A')

            response = f"📝 **Review Analysis: {name}**\n\n"
            response += f"⭐ Overall Rating: {rating}/5 ({rating_count} reviews)\n\n"

            # Extract reviews from content
            content = doc.page_content

            if "Customer Reviews:" in content:
                reviews_section = content.split("Customer Reviews:")[1]
                reviews = [r.strip() for r in reviews_section.split("-") if r.strip()][:5]

                if reviews:
                    response += "**Customer Feedback:**\n"
                    for review in reviews:
                        if len(review) > 20:
                            response += f"  • {review[:200]}...\n"

            # Sentiment indicators
            content_lower = content.lower()
            positives = []
            negatives = []

            if "good" in content_lower or "great" in content_lower or "excellent" in content_lower:
                positives.append("Positive feedback on quality")
            if "fast" in content_lower and "charging" in content_lower:
                positives.append("Fast charging appreciated")
            if "durable" in content_lower or "strong" in content_lower:
                positives.append("Durable construction noted")
            if "value" in content_lower and "money" in content_lower:
                positives.append("Good value for money")

            if "issue" in content_lower or "problem" in content_lower:
                negatives.append("Some reported issues")
            if "not working" in content_lower or "stopped" in content_lower:
                negatives.append("Durability concerns from some users")

            if positives:
                response += f"\n✅ **Positives:**\n"
                for p in positives:
                    response += f"   • {p}\n"

            if negatives:
                response += f"\n⚠️ **Consider:**\n"
                for n in negatives:
                    response += f"   • {n}\n"

            # Recommendation
            rating_float = float(rating) if rating != 'N/A' else 0
            if rating_float >= 4.2:
                response += "\n✅ **Recommendation:** Highly recommended based on customer reviews!"
            elif rating_float >= 4.0:
                response += "\n👍 **Recommendation:** Good choice with solid reviews."
            elif rating_float >= 3.5:
                response += "\n⚠️ **Recommendation:** Consider your needs - mixed reviews."
            else:
                response += "\n❌ **Recommendation:** May want to explore alternatives."

            return response

        except Exception as e:
            return f"Error analyzing reviews: {str(e)}"


def main():
    """Demo the simple recommendation system"""
    print("="*70)
    print("Simple RAG-based Recommendation System (No OpenAI API Required)")
    print("="*70)

    # Initialize RAG
    from dotenv import load_dotenv
    load_dotenv()

    print("\n1. Loading RAG system...")
    rag = RAGSystem(persist_directory="./vectorstore/chroma_db", use_openai=False)
    rag.load_vectorstore()
    print("✅ RAG system loaded\n")

    # Initialize recommender
    recommender = SimpleRecommendationSystem(rag)

    # Demo queries
    print("="*70)
    print("Demo 1: Product Search")
    print("="*70)
    print(recommender.search_products("USB cable fast charging", k=3))

    print("\n" + "="*70)
    print("Demo 2: Recommendations")
    print("="*70)
    print(recommender.recommend_products("charger cable with above 4 star rating"))

    print("\n" + "="*70)
    print("Demo 3: Review Analysis")
    print("="*70)
    print(recommender.analyze_reviews("boAt cable"))


if __name__ == "__main__":
    main()
