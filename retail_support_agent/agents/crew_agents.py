"""
Crew AI Agent System for Intelligent Retail Customer Support
Implements multi-agent collaboration with specialized roles
"""

from crewai import Agent, Task, Crew, Process
from langchain.tools import tool  # Use LangChain's tool decorator (more stable)
from langchain_openai import ChatOpenAI
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rag_system import RAGSystem


class RetailSupportCrew:
    """Multi-agent system for retail customer support"""

    def __init__(
        self,
        rag_system: RAGSystem,
        model_name: str = "gpt-3.5-turbo",  # Using gpt-3.5-turbo (available and affordable)
        temperature: float = 0.7
    ):
        """Initialize the crew with RAG system"""
        self.rag_system = rag_system
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        # Initialize agents (without custom tools for now to avoid compatibility issues)
        self.customer_support_agent = self.create_customer_support_agent()
        self.product_expert_agent = self.create_product_expert_agent()
        self.review_analyzer_agent = self.create_review_analyzer_agent()
        self.recommendation_agent = self.create_recommendation_agent()

    def setup_tools(self):
        """Setup custom tools for agents"""

        @tool
        def search_products(query: str) -> str:
            """Search for products based on customer query. Use this tool to find relevant products in the catalog."""
            try:
                results = self.rag_system.similarity_search(query, k=3)
                output = []
                for i, doc in enumerate(results, 1):
                    metadata = doc.metadata
                    output.append(f"""
Product {i}:
Name: {metadata.get('product_name', 'N/A')}
Category: {metadata.get('category', 'N/A')}
Price: {metadata.get('price', 'N/A')}
Rating: {metadata.get('rating', 'N/A')}/5
Link: {metadata.get('product_link', 'N/A')}

Details:
{doc.page_content[:300]}...
""")
                return "\n".join(output)
            except Exception as e:
                return f"Error searching products: {str(e)}"

        @tool
        def analyze_reviews(product_query: str) -> str:
            """Analyze customer reviews for a specific product. Use this to understand customer sentiment and feedback."""
            try:
                results = self.rag_system.similarity_search(product_query, k=1)
                if results:
                    doc = results[0]
                    # Extract review section
                    content = doc.page_content
                    if "Customer Reviews:" in content:
                        reviews_section = content.split("Customer Reviews:")[1]
                        return f"Customer Reviews Analysis:\n{reviews_section[:500]}..."
                    return "No reviews found for this product."
                return "Product not found."
            except Exception as e:
                return f"Error analyzing reviews: {str(e)}"

        @tool
        def compare_products(product_names: str) -> str:
            """Compare multiple products based on features, price, and ratings. Provide product names separated by commas."""
            try:
                products = [p.strip() for p in product_names.split(',')]
                comparison = []

                for product_name in products[:3]:  # Limit to 3 products
                    results = self.rag_system.get_product_by_name(product_name, k=1)
                    if results:
                        doc = results[0]
                        metadata = doc.metadata
                        comparison.append(f"""
{metadata.get('product_name', 'N/A')}:
- Price: {metadata.get('price', 'N/A')}
- Rating: {metadata.get('rating', 'N/A')}/5
- Category: {metadata.get('category', 'N/A')}
""")

                return "Product Comparison:\n" + "\n".join(comparison)
            except Exception as e:
                return f"Error comparing products: {str(e)}"

        self.search_products_tool = search_products
        self.analyze_reviews_tool = analyze_reviews
        self.compare_products_tool = compare_products

    def create_customer_support_agent(self) -> Agent:
        """Create the main customer support agent"""
        return Agent(
            role='Senior Customer Support Specialist',
            goal='Provide exceptional customer service and resolve customer inquiries efficiently',
            backstory="""You are an experienced customer support specialist with 10+ years
            in retail e-commerce. You excel at understanding customer needs, providing accurate
            information, and ensuring customer satisfaction. You are empathetic, professional,
            and detail-oriented. You always verify information before providing answers and
            escalate complex issues to specialized agents when needed.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )

    def create_product_expert_agent(self) -> Agent:
        """Create the product expert agent"""
        return Agent(
            role='Product Specialist and Technical Expert',
            goal='Provide detailed product information, specifications, and technical guidance',
            backstory="""You are a product expert with deep knowledge of electronics,
            accessories, and technology products. You can explain technical specifications,
            compare features, and help customers understand product capabilities. You stay
            updated on product trends and can provide insights on product quality,
            compatibility, and use cases.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def create_review_analyzer_agent(self) -> Agent:
        """Create the review analysis agent"""
        return Agent(
            role='Customer Feedback Analyst',
            goal='Analyze customer reviews and provide sentiment-based insights',
            backstory="""You are an expert in analyzing customer feedback and reviews.
            You can identify patterns in customer satisfaction, highlight common issues,
            and extract valuable insights from reviews. You help customers make informed
            decisions by presenting balanced views from actual user experiences.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def create_recommendation_agent(self) -> Agent:
        """Create the product recommendation agent"""
        return Agent(
            role='Personal Shopping Assistant',
            goal='Provide personalized product recommendations based on customer needs',
            backstory="""You are a personal shopping assistant with expertise in matching
            customer requirements with the right products. You consider budget, preferences,
            ratings, and reviews to suggest the best options. You can identify upsell
            opportunities while ensuring customer satisfaction.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )

    def _get_rag_context(self, query: str, k: int = 3) -> str:
        """Helper method to retrieve context from RAG system"""
        try:
            results = self.rag_system.similarity_search(query, k=k)
            context_parts = []
            for i, doc in enumerate(results, 1):
                metadata = doc.metadata
                context_parts.append(f"""
Product {i}: {metadata.get('product_name', 'N/A')}
Category: {metadata.get('category', 'N/A')}
Price: {metadata.get('price', 'N/A')}
Rating: {metadata.get('rating', 'N/A')}/5
Details: {doc.page_content[:400]}...
""")
            return "\n".join(context_parts)
        except Exception as e:
            return f"Unable to retrieve product information: {str(e)}"

    def handle_customer_query(self, query: str, query_type: str = "general") -> str:
        """
        Handle different types of customer queries

        Args:
            query: Customer's question or request
            query_type: Type of query (general, product_search, review, comparison, recommendation)

        Returns:
            Response from the agent crew
        """

        if query_type == "product_search":
            # Retrieve context from RAG system
            context = self._get_rag_context(query, k=3)

            task = Task(
                description=f"""
                Customer Query: {query}

                Available Product Information:
                {context}

                Task: Based on the product information above, provide a helpful response including:
                1. Product names and descriptions
                2. Pricing information
                3. Ratings and review counts
                4. Key features
                5. Your recommendations

                Be specific and helpful in your response.
                """,
                agent=self.product_expert_agent,
                expected_output="Detailed product information with at least 2-3 relevant products"
            )

            crew = Crew(
                agents=[self.product_expert_agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential
            )

        elif query_type == "review_analysis":
            # Retrieve context from RAG system
            context = self._get_rag_context(query, k=2)

            task = Task(
                description=f"""
                Customer Query: {query}

                Product Information and Reviews:
                {context}

                Task: Based on the product information and reviews above, analyze:
                1. Overall sentiment (positive/negative/mixed)
                2. Common praises from customers
                3. Common complaints or issues
                4. Recommendation based on reviews

                Be balanced and honest in your analysis.
                """,
                agent=self.review_analyzer_agent,
                expected_output="Comprehensive review analysis with sentiment and insights"
            )

            crew = Crew(
                agents=[self.review_analyzer_agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential
            )

        elif query_type == "comparison":
            # Retrieve context from RAG system
            context = self._get_rag_context(query, k=4)

            task = Task(
                description=f"""
                Customer Query: {query}

                Products to Compare:
                {context}

                Task: Based on the product information above, provide:
                1. Side-by-side feature comparison
                2. Price comparison
                3. Rating comparison
                4. Pros and cons of each
                5. Recommendation based on different use cases

                Be objective and thorough.
                """,
                agent=self.product_expert_agent,
                expected_output="Detailed product comparison with clear recommendations"
            )

            crew = Crew(
                agents=[self.product_expert_agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential
            )

        elif query_type == "recommendation":
            # Retrieve context from RAG system
            context = self._get_rag_context(query, k=5)

            task = Task(
                description=f"""
                Customer Query: {query}

                Available Products:
                {context}

                Task: Based on the customer's needs and available products, provide personalized recommendations:
                1. Identify 2-3 suitable products from the list above
                2. Explain why each product matches their needs
                3. Compare prices and features
                4. Provide a final recommendation

                Be helpful and consider customer's budget and requirements.
                """,
                agent=self.recommendation_agent,
                expected_output="Personalized product recommendations with justification"
            )

            crew = Crew(
                agents=[self.recommendation_agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential
            )

        else:  # general query - multi-agent collaboration
            # Retrieve context from RAG system
            context = self._get_rag_context(query, k=3)

            # Create tasks for multi-agent workflow
            support_task = Task(
                description=f"""
                Customer Query: {query}

                Available Product Information:
                {context}

                Task: Understand the customer's query and provide helpful assistance based on the available product information.
                """,
                agent=self.customer_support_agent,
                expected_output="Helpful response to customer query with product information"
            )

            expert_task = Task(
                description=f"""
                Customer Query: {query}

                Product Information:
                {context}

                Task: Provide expert product information and technical details based on the available information.
                """,
                agent=self.product_expert_agent,
                expected_output="Expert product information and technical guidance"
            )

            crew = Crew(
                agents=[
                    self.customer_support_agent,
                    self.product_expert_agent
                ],
                tasks=[support_task, expert_task],
                verbose=True,
                process=Process.sequential
            )

        # Execute crew
        result = crew.kickoff()
        return str(result)


def main():
    """Test the crew system"""
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize RAG system
    print("Loading RAG system...")
    rag = RAGSystem(persist_directory="../vectorstore/chroma_db")
    rag.load_vectorstore()

    # Initialize crew
    print("\nInitializing Crew AI agents...")
    crew_system = RetailSupportCrew(rag)

    # Test queries
    test_cases = [
        ("What are the best USB cables under 500 rupees?", "product_search"),
        ("Tell me about customer reviews for boAt cables", "review_analysis"),
        ("I need a good charging cable for my iPhone", "recommendation")
    ]

    print("\n" + "="*70)
    print("Testing Crew AI Multi-Agent System")
    print("="*70)

    for query, query_type in test_cases:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"Type: {query_type}")
        print("="*70)

        response = crew_system.handle_customer_query(query, query_type)
        print(f"\nResponse:\n{response}")
        print("\n")


if __name__ == "__main__":
    main()
