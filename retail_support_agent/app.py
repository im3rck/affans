"""
Streamlit Demo Interface for Intelligent Retail Customer Support Agent
Multi-page application with agent interaction, analytics, and evaluation
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.rag_system import RAGSystem
from agents.crew_agents import RetailSupportCrew
from agents.prompt_templates import PromptTemplates, PromptStrategy
from dotenv import load_dotenv
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Retail Support Agent",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .agent-message {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'crew_system' not in st.session_state:
    st.session_state.crew_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False


@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system (cached)"""
    with st.spinner("Loading RAG system..."):
        # Use local embeddings by default (no API key needed)
        rag = RAGSystem(persist_directory="./vectorstore/chroma_db", use_openai=False)
        try:
            rag.load_vectorstore()
            return rag, True
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            st.info("Please run data preprocessing first: `python utils/data_preprocessor.py`")
            return None, False


@st.cache_resource
def initialize_crew_system(_rag_system):
    """Initialize Crew AI system (cached)"""
    with st.spinner("Initializing AI agents..."):
        try:
            crew = RetailSupportCrew(_rag_system)
            return crew, True
        except Exception as e:
            st.error(f"Error initializing agents: {e}")
            return None, False


def main():
    """Main application"""

    # Sidebar
    with st.sidebar:
        st.title("🛍️ Retail Support Agent")
        st.markdown("---")

        # Page selection
        page = st.radio(
            "Navigation",
            ["💬 Chat Support", "📊 Analytics", "🔍 Product Search", "⚙️ Settings"]
        )

        st.markdown("---")

        # System status
        st.subheader("System Status")

        if not st.session_state.system_initialized:
            if st.button("Initialize System"):
                rag, rag_success = initialize_rag_system()
                if rag_success:
                    st.session_state.rag_system = rag
                    crew, crew_success = initialize_crew_system(rag)
                    if crew_success:
                        st.session_state.crew_system = crew
                        st.session_state.system_initialized = True
                        st.success("System initialized successfully!")
                        st.rerun()
        else:
            st.success("✅ System Active")
            stats = st.session_state.rag_system.get_statistics()
            st.metric("Documents", stats.get('total_documents', 0))
            st.metric("Chat Sessions", len(st.session_state.chat_history))

        st.markdown("---")
        st.caption("Powered by CrewAI, RAG, and RAGAS")

    # Main content area
    if not st.session_state.system_initialized:
        show_welcome_page()
    else:
        if page == "💬 Chat Support":
            show_chat_page()
        elif page == "📊 Analytics":
            show_analytics_page()
        elif page == "🔍 Product Search":
            show_search_page()
        elif page == "⚙️ Settings":
            show_settings_page()


def show_welcome_page():
    """Show welcome/initialization page"""
    st.markdown('<div class="main-header">🛍️ Intelligent Retail Customer Support Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Multi-Agent System with RAG & Fine-Tuning</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🤖 Multi-Agent System")
        st.write("- Customer Support Agent")
        st.write("- Product Expert")
        st.write("- Review Analyzer")
        st.write("- Recommendation Engine")

    with col2:
        st.markdown("### 🔍 Advanced RAG")
        st.write("- Vector Database (ChromaDB)")
        st.write("- Semantic Search")
        st.write("- Context Retrieval")
        st.write("- Real-time Updates")

    with col3:
        st.markdown("### 📈 Evaluation")
        st.write("- RAGAS Metrics")
        st.write("- Faithfulness Scoring")
        st.write("- Answer Relevancy")
        st.write("- Context Precision")

    st.markdown("---")

    st.info("👈 Click 'Initialize System' in the sidebar to get started!")

    # Features showcase
    st.markdown("### ✨ Key Features")

    features = [
        ("🎯", "Intelligent Query Routing", "Automatically routes queries to specialized agents"),
        ("💡", "Context-Aware Responses", "Uses RAG to provide accurate, data-backed answers"),
        ("⚡", "Fast Performance", "Optimized vector search and caching"),
        ("📊", "Analytics Dashboard", "Track performance and customer insights"),
        ("🔧", "Fine-Tuned Models", "Custom-trained on retail customer support data"),
        ("✅", "Quality Assurance", "RAGAS evaluation ensures high-quality responses")
    ]

    col1, col2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)


def show_chat_page():
    """Show chat interface"""
    st.markdown('<div class="main-header">💬 Customer Support Chat</div>', unsafe_allow_html=True)

    # Query type selection
    col1, col2 = st.columns([3, 1])

    with col1:
        query_type = st.selectbox(
            "Query Type",
            ["general", "product_search", "review_analysis", "comparison", "recommendation"],
            format_func=lambda x: {
                "general": "General Inquiry",
                "product_search": "Product Search",
                "review_analysis": "Review Analysis",
                "comparison": "Product Comparison",
                "recommendation": "Get Recommendations"
            }[x]
        )

    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Display chat history
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message user-message">👤 **You**: {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message agent-message">🤖 **Agent**: {message["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    st.markdown("---")

    # Example queries
    with st.expander("💡 Example Queries"):
        examples = {
            "product_search": [
                "What are the best USB cables under 500 rupees?",
                "Show me charging cables with good ratings"
            ],
            "review_analysis": [
                "What do customers say about boAt cables?",
                "Are Ambrane cables durable?"
            ],
            "comparison": [
                "Compare boAt and Ambrane cables",
                "Which is better: Wayona or pTron cable?"
            ],
            "recommendation": [
                "I need a charging cable for my iPhone",
                "Recommend a durable USB-C cable for laptop"
            ]
        }

        for ex in examples.get(query_type, []):
            if st.button(ex, key=f"example_{ex}"):
                handle_query(ex, query_type)
                st.rerun()

    # Chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your message:",
            placeholder="Ask me anything about products...",
            key="user_input"
        )
        submit = st.form_submit_button("Send")

        if submit and user_input:
            handle_query(user_input, query_type)
            st.rerun()


def handle_query(query: str, query_type: str):
    """Handle user query"""
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': query
    })

    try:
        # Get response from crew system
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.crew_system.handle_customer_query(
                query,
                query_type
            )

        # Add agent response to history
        st.session_state.chat_history.append({
            'role': 'agent',
            'content': response
        })

    except Exception as e:
        st.error(f"Error processing query: {e}")
        st.session_state.chat_history.append({
            'role': 'agent',
            'content': "I apologize, but I encountered an error. Please try again."
        })


def show_search_page():
    """Show product search page"""
    st.markdown('<div class="main-header">🔍 Product Search</div>', unsafe_allow_html=True)

    # Search input
    search_query = st.text_input("Search products:", placeholder="e.g., USB cable, charging cable, boAt")

    col1, col2, col3 = st.columns(3)
    with col1:
        num_results = st.slider("Number of results", 1, 10, 5)
    with col2:
        min_rating = st.slider("Minimum rating", 0.0, 5.0, 3.0, 0.1)

    if search_query:
        with st.spinner("Searching..."):
            results = st.session_state.rag_system.similarity_search_with_score(
                search_query,
                k=num_results
            )

            st.markdown(f"### Found {len(results)} products")

            for i, (doc, score) in enumerate(results, 1):
                with st.expander(f"📦 {doc.metadata.get('product_name', 'Unknown')} (Relevance: {score:.3f})"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Price", doc.metadata.get('price', 'N/A'))
                    with col2:
                        st.metric("Rating", f"{doc.metadata.get('rating', 'N/A')}/5")
                    with col3:
                        st.metric("Category", doc.metadata.get('category', 'N/A')[:20] + "...")

                    st.markdown("**Product Details:**")
                    st.write(doc.page_content[:500] + "...")

                    if doc.metadata.get('product_link'):
                        st.markdown(f"[View Product]({doc.metadata['product_link']})")


def show_analytics_page():
    """Show analytics dashboard"""
    st.markdown('<div class="main-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

    # Load statistics
    try:
        with open('./data/processed/dataset_stats.json', 'r') as f:
            stats = json.load(f)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Products", stats.get('total_products', 0))
        with col2:
            st.metric("Training Examples", stats.get('total_training_examples', 0))
        with col3:
            st.metric("Categories", stats.get('categories', 0))
        with col4:
            st.metric("Avg Rating", f"{stats.get('avg_rating', 0):.2f}/5")

        st.markdown("---")

        # Chat analytics
        if st.session_state.chat_history:
            st.subheader("📈 Chat Session Analytics")

            chat_df = pd.DataFrame(st.session_state.chat_history)

            col1, col2 = st.columns(2)

            with col1:
                # Message count
                fig = px.pie(
                    values=[
                        len(chat_df[chat_df['role'] == 'user']),
                        len(chat_df[chat_df['role'] == 'agent'])
                    ],
                    names=['User Messages', 'Agent Responses'],
                    title='Message Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.metric("Total Messages", len(chat_df))
                st.metric("User Queries", len(chat_df[chat_df['role'] == 'user']))
                st.metric("Agent Responses", len(chat_df[chat_df['role'] == 'agent']))

        # RAGAS evaluation results
        st.markdown("---")
        st.subheader("🎯 RAGAS Evaluation Metrics")

        try:
            ragas_df = pd.read_csv('./data/ragas_results.csv')

            metric_cols = [col for col in ragas_df.columns if col not in ['question', 'answer', 'contexts', 'ground_truth']]

            if metric_cols:
                fig = go.Figure()

                for metric in metric_cols:
                    fig.add_trace(go.Box(
                        y=ragas_df[metric],
                        name=metric.replace('_', ' ').title()
                    ))

                fig.update_layout(
                    title='RAGAS Metrics Distribution',
                    yaxis_title='Score',
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

                # Metric averages
                st.markdown("### Average Scores")
                cols = st.columns(len(metric_cols))
                for i, metric in enumerate(metric_cols):
                    with cols[i]:
                        avg_score = ragas_df[metric].mean()
                        st.metric(
                            metric.replace('_', ' ').title(),
                            f"{avg_score:.3f}",
                            delta=f"{(avg_score - 0.7):.3f}" if avg_score >= 0.7 else None
                        )
        except FileNotFoundError:
            st.info("Run RAGAS evaluation to see metrics: `python utils/ragas_evaluation.py`")

    except FileNotFoundError:
        st.warning("No analytics data available. Please run data preprocessing first.")


def show_settings_page():
    """Show settings page"""
    st.markdown('<div class="main-header">⚙️ Settings</div>', unsafe_allow_html=True)

    # System information
    st.subheader("System Information")

    if st.session_state.rag_system:
        stats = st.session_state.rag_system.get_statistics()

        info_data = {
            "Component": ["Vector Store", "Collection", "Embedding Model", "Total Documents"],
            "Value": [
                "ChromaDB",
                stats.get('collection_name', 'N/A'),
                stats.get('embedding_model', 'N/A'),
                stats.get('total_documents', 0)
            ]
        }

        st.table(pd.DataFrame(info_data))

    st.markdown("---")

    # Prompt strategy settings
    st.subheader("Prompt Engineering Strategy")

    strategy = st.selectbox(
        "Select Prompt Strategy",
        options=list(PromptStrategy),
        format_func=lambda x: x.value.replace('_', ' ').title()
    )

    st.info(f"Currently using: **{strategy.value}** prompting strategy")

    # Test prompt
    with st.expander("🧪 Test Prompt Template"):
        test_query = st.text_input("Test Query", "What are the best USB cables?")
        test_context = st.text_area("Context", "boAt cable: ₹329, 4.2 rating\nAmbrane cable: ₹199, 4.0 rating")

        if st.button("Generate Prompt"):
            prompt = PromptTemplates.get_prompt(strategy, test_query, test_context)
            st.code(prompt, language="text")

    st.markdown("---")

    # Data management
    st.subheader("Data Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Refresh Vector Store"):
            st.info("Refreshing vector store...")
            st.session_state.rag_system.load_vectorstore()
            st.success("Vector store refreshed!")

    with col2:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")


if __name__ == "__main__":
    main()
