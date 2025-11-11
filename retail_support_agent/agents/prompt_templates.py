"""
Advanced Prompt Engineering Templates
Implements various prompting techniques: Few-shot, Chain-of-Thought, ReAct, etc.
"""

from typing import List, Dict, Optional
from enum import Enum


class PromptStrategy(Enum):
    """Different prompt engineering strategies"""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    INSTRUCTION_FOLLOWING = "instruction_following"


class PromptTemplates:
    """Collection of engineered prompt templates"""

    @staticmethod
    def get_system_prompt(role: str = "customer_support") -> str:
        """Get role-specific system prompts"""

        prompts = {
            "customer_support": """You are an expert retail customer support agent with the following qualities:

EXPERTISE:
- 10+ years of experience in e-commerce customer support
- Deep knowledge of products, policies, and best practices
- Excellent communication and problem-solving skills

GUIDELINES:
1. Always be polite, professional, and empathetic
2. Provide accurate information backed by data
3. If unsure, acknowledge it and offer to find more information
4. Suggest alternatives when primary options aren't available
5. Focus on customer satisfaction and resolution

RESPONSE FORMAT:
- Start with acknowledging the customer's query
- Provide clear, structured information
- Include relevant product details (price, ratings, features)
- End with a helpful question or next step suggestion

Remember: Your goal is to help customers make informed decisions and resolve their concerns efficiently.""",

            "product_expert": """You are a product specialist with deep technical knowledge.

EXPERTISE:
- Comprehensive understanding of product specifications
- Ability to compare and contrast different products
- Knowledge of compatibility, use cases, and best practices

YOUR APPROACH:
1. Analyze product features in detail
2. Provide objective comparisons
3. Explain technical terms in simple language
4. Consider customer's specific needs and context
5. Highlight both advantages and limitations

RESPONSE STRUCTURE:
- Product overview
- Key specifications
- Comparison with alternatives (if relevant)
- Recommendation based on use case
- Additional considerations""",

            "review_analyzer": """You are a customer feedback analyst specializing in sentiment analysis.

YOUR ROLE:
- Analyze customer reviews objectively
- Identify patterns in feedback
- Extract actionable insights
- Present balanced perspectives

ANALYSIS FRAMEWORK:
1. Overall Sentiment (Positive/Neutral/Negative)
2. Common Praises (Top 3-5 points)
3. Common Complaints (Top 3-5 points)
4. Specific Use Cases mentioned
5. Value for Money perception
6. Final Recommendation

TONE:
- Objective and data-driven
- Balanced (acknowledge both positives and negatives)
- Helpful and informative""",

            "recommendation": """You are a personal shopping assistant dedicated to finding the perfect products.

YOUR MISSION:
- Understand customer needs deeply
- Match requirements with available products
- Consider budget, quality, and reviews
- Provide personalized recommendations

RECOMMENDATION PROCESS:
1. Clarify customer requirements
2. Search relevant products
3. Filter based on criteria (price, rating, features)
4. Compare top options
5. Present ranked recommendations with reasoning

CRITERIA TO CONSIDER:
- Budget constraints
- Primary use case
- Quality indicators (ratings, reviews)
- Brand reputation
- Value for money
- Special features needed"""
        }

        return prompts.get(role, prompts["customer_support"])

    @staticmethod
    def zero_shot_prompt(query: str, context: str = "") -> str:
        """Zero-shot prompting"""
        template = """Answer the following customer query based on the provided context.

Context:
{context}

Customer Query: {query}

Response:"""

        return template.format(context=context, query=query)

    @staticmethod
    def few_shot_prompt(query: str, context: str = "") -> str:
        """Few-shot prompting with examples"""
        template = """Answer the customer query using the provided context. Here are some examples:

Example 1:
Query: "What's the price of boAt cable?"
Response: "The boAt Deuce USB 300 cable is currently priced at ₹329, down from ₹699, giving you a 53% discount. It has a 4.2/5 rating from 94,363 customers and comes with a 2-year warranty."

Example 2:
Query: "Is the Wayona cable durable?"
Response: "Yes, the Wayona cable is designed for durability. According to customer reviews, it features a nylon braided design with premium aluminum housing and has passed 10,000+ bending tests. Customers specifically mention: 'Looks durable,' 'Good quality,' and 'Cable is sturdy enough.'"

Example 3:
Query: "Which cable supports fast charging?"
Response: "Several cables support fast charging:\n1. boAt Deuce USB 300 - 3A fast charging\n2. Ambrane Unbreakable - 3A/60W fast charging\n3. Portronics Konnect L - Fast charging supported\n\nThe Ambrane cable offers the highest power at 60W, making it suitable for laptops and tablets too."

Now, answer this query:

Context:
{context}

Customer Query: {query}

Response:"""

        return template.format(context=context, query=query)

    @staticmethod
    def chain_of_thought_prompt(query: str, context: str = "") -> str:
        """Chain-of-Thought prompting for complex reasoning"""
        template = """Answer the customer query using step-by-step reasoning.

Context:
{context}

Customer Query: {query}

Let's think through this step by step:

Step 1: Understand the customer's needs
- What is the customer looking for?
- What are their key requirements?
- What constraints do they have (budget, features, etc.)?

Step 2: Analyze available options
- What products match the criteria?
- What are the key features of each?
- How do they compare?

Step 3: Evaluate based on criteria
- Price comparison
- Rating and review analysis
- Feature comparison
- Value for money assessment

Step 4: Formulate recommendation
- Best option(s) based on analysis
- Reasoning for recommendation
- Any caveats or considerations

Final Response:"""

        return template.format(context=context, query=query)

    @staticmethod
    def react_prompt(query: str, context: str = "") -> str:
        """ReAct (Reasoning + Acting) prompting"""
        template = """Answer the customer query using the ReAct framework: Thought → Action → Observation → Response

Context:
{context}

Customer Query: {query}

Thought: What information do I need to answer this query effectively?

Action: [Specify what you would search for or analyze]

Observation: [Analyze the context and extract relevant information]

Thought: Based on the observation, what's the best way to help the customer?

Action: [Formulate a helpful response]

Response: [Provide the final answer to the customer]"""

        return template.format(context=context, query=query)

    @staticmethod
    def instruction_following_prompt(query: str, context: str, instructions: List[str]) -> str:
        """Detailed instruction-following prompt"""
        instructions_text = "\n".join([f"{i+1}. {inst}" for i, inst in enumerate(instructions)])

        template = """Follow these instructions carefully to answer the customer query:

INSTRUCTIONS:
{instructions}

CONTEXT:
{context}

CUSTOMER QUERY:
{query}

YOUR RESPONSE (following all instructions above):"""

        return template.format(
            instructions=instructions_text,
            context=context,
            query=query
        )

    @staticmethod
    def product_comparison_prompt(products: List[str], criteria: List[str]) -> str:
        """Structured product comparison prompt"""
        products_list = "\n".join([f"- {p}" for p in products])
        criteria_list = "\n".join([f"- {c}" for c in criteria])

        template = """Compare the following products based on the specified criteria:

PRODUCTS TO COMPARE:
{products}

COMPARISON CRITERIA:
{criteria}

Provide a structured comparison table followed by:
1. Summary of key differences
2. Pros and cons of each product
3. Recommendation based on different use cases
4. Final verdict

COMPARISON:"""

        return template.format(products=products_list, criteria=criteria_list)

    @staticmethod
    def sentiment_analysis_prompt(product_name: str, reviews: str) -> str:
        """Prompt for analyzing product reviews"""
        template = """Analyze the customer reviews for: {product_name}

REVIEWS:
{reviews}

Provide a comprehensive analysis including:

1. OVERALL SENTIMENT SCORE: (1-5 scale)
   - Calculate based on review content

2. POSITIVE ASPECTS (Top 3-5):
   - List what customers love about this product
   - Include specific quotes if available

3. NEGATIVE ASPECTS (Top 3-5):
   - List common complaints or issues
   - Include specific quotes if available

4. KEY THEMES:
   - Identify recurring topics in reviews

5. CUSTOMER PROFILES:
   - Who is this product best suited for?

6. VALUE FOR MONEY:
   - Is the product worth its price based on reviews?

7. RECOMMENDATION:
   - Would you recommend this product? Why or why not?
   - What type of customer would benefit most?

ANALYSIS:"""

        return template.format(product_name=product_name, reviews=reviews)

    @staticmethod
    def contextual_qa_prompt(
        query: str,
        context: str,
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """Prompt with conversation history for contextual understanding"""

        history_text = ""
        if chat_history:
            history_text = "CONVERSATION HISTORY:\n"
            for turn in chat_history[-3:]:  # Last 3 turns
                history_text += f"Customer: {turn.get('query', '')}\n"
                history_text += f"Agent: {turn.get('response', '')}\n\n"

        template = """{history}

CONTEXT:
{context}

CURRENT QUERY: {query}

Consider the conversation history to provide a contextually relevant response.
If this is a follow-up question, reference previous information appropriately.

RESPONSE:"""

        return template.format(
            history=history_text,
            context=context,
            query=query
        )

    @staticmethod
    def get_prompt(
        strategy: PromptStrategy,
        query: str,
        context: str = "",
        **kwargs
    ) -> str:
        """Get prompt based on strategy"""

        strategy_map = {
            PromptStrategy.ZERO_SHOT: PromptTemplates.zero_shot_prompt,
            PromptStrategy.FEW_SHOT: PromptTemplates.few_shot_prompt,
            PromptStrategy.CHAIN_OF_THOUGHT: PromptTemplates.chain_of_thought_prompt,
            PromptStrategy.REACT: PromptTemplates.react_prompt,
        }

        prompt_func = strategy_map.get(strategy, PromptTemplates.zero_shot_prompt)
        return prompt_func(query, context)


# Example usage and testing
if __name__ == "__main__":
    # Test different prompt strategies
    test_query = "Which USB cable should I buy for fast charging my phone?"
    test_context = """
    Product 1: boAt Deuce USB 300 - ₹329, 4.2 rating, 3A fast charging
    Product 2: Ambrane Unbreakable - ₹199, 4.0 rating, 3A/60W fast charging
    Product 3: Wayona Nylon Braided - ₹399, 4.2 rating, Fast charging supported
    """

    print("="*70)
    print("PROMPT ENGINEERING EXAMPLES")
    print("="*70)

    strategies = [
        PromptStrategy.ZERO_SHOT,
        PromptStrategy.FEW_SHOT,
        PromptStrategy.CHAIN_OF_THOUGHT,
        PromptStrategy.REACT
    ]

    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy.value.upper()}")
        print("="*70)
        prompt = PromptTemplates.get_prompt(strategy, test_query, test_context)
        print(prompt)
        print("\n")
