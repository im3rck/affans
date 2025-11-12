"""
Intelligent Retail System - Streamlit Web Interface
Advanced RAG + Hybrid ML + LLM Integration (No Crew AI)
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.intelligent_system import IntelligentRetailSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Intelligent Retail System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the Intelligent Retail System (cached)"""
    with st.spinner("🚀 Initializing Intelligent Retail System..."):
        try:
            # Use local embeddings by default (no API key needed for embeddings)
            system = IntelligentRetailSystem(
                vectorstore_path="./vectorstore/chroma_db",
                data_path="./data/processed/processed_products.csv",
                use_openai_embeddings=False,  # FREE local embeddings
                llm_model="gpt-3.5-turbo"
            )
            return system, True
        except Exception as e:
            st.error(f"❌ Error initializing system: {e}")
            return None, False


def render_header():
    """Render page header"""
    st.markdown('<h1 class="main-header">🛒 Intelligent Retail System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Advanced RAG + Hybrid ML + LLM Integration</p>',
        unsafe_allow_html=True
    )

    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔍 **Advanced RAG**\nQuery Rewriting • Hybrid Search • Reranking")
    with col2:
        st.success("🤖 **Hybrid ML + GenAI**\nClustering • Classification • LLM Insights")
    with col3:
        st.warning("💬 **Intelligent Q&A**\nNatural Language Interface")


def render_sidebar(system):
    """Render sidebar with system info"""
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=150)
        st.title("System Info")

        st.markdown("---")

        # System Status
        st.subheader("🟢 Status")
        if system:
            st.success("System Online")
            st.caption(f"Model: gpt-3.5-turbo")
            st.caption(f"Embeddings: HuggingFace (Local)")
        else:
            st.error("System Offline")

        st.markdown("---")

        # Features
        st.subheader("✨ Features")
        features = [
            "🔍 Advanced RAG Search",
            "🎯 Smart Recommendations",
            "📊 Market Analysis",
            "💬 Intelligent Q&A",
            "🤖 ML Insights"
        ]
        for feature in features:
            st.caption(feature)

        st.markdown("---")

        # About
        st.subheader("ℹ️ About")
        st.caption("""
        This system combines:
        - **Advanced RAG**: Query rewriting, hybrid search, reranking
        - **Hybrid ML**: Clustering, classification with LLM interpretation
        - **LLM Integration**: GPT-3.5-turbo for natural language
        """)

        st.markdown("---")

        # Credits
        st.caption("Built for AI Bootcamp")
        st.caption("Advanced RAG + ML + GenAI")


def tab_search(system):
    """Product Search Tab"""
    st.header("🔍 Advanced Product Search")

    st.markdown("""
    Search for products using our **Advanced RAG** system with:
    - ✨ Query Rewriting (expands and clarifies your search)
    - 🔎 Hybrid Search (semantic + keyword matching)
    - 🎯 Reranking (most relevant results first)
    """)

    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search for products:",
            placeholder="e.g., affordable USB cable with fast charging",
            label_visibility="collapsed"
        )
    with col2:
        num_results = st.selectbox("Results", [3, 5, 10], index=0)

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        use_advanced = st.checkbox("Use Advanced RAG Pipeline", value=True,
                                   help="Enable query rewriting, hybrid search, and reranking")

    # Search button
    if st.button("🔍 Search", type="primary"):
        if not query:
            st.warning("Please enter a search query")
        else:
            with st.spinner("Searching with Advanced RAG..."):
                try:
                    result = system.search_products(
                        query=query,
                        k=num_results,
                        use_advanced=use_advanced
                    )
                    st.markdown("### Results")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Search failed: {e}")

    # Example queries
    st.markdown("---")
    st.markdown("**💡 Example Searches:**")
    examples = [
        "affordable USB cable with good reviews",
        "premium charging cable under 500 rupees",
        "fast charging cable with warranty",
        "durable phone charger cable"
    ]
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"Try: {example}", key=f"ex_search_{i}"):
                st.session_state['search_query'] = example
                st.rerun()


def tab_recommendations(system):
    """Recommendations Tab"""
    st.header("🎯 Smart Recommendations")

    st.markdown("""
    Get intelligent product recommendations based on:
    - 🔍 Advanced RAG product search
    - 📊 ML-powered market insights
    - 💡 LLM analysis and reasoning
    """)

    # Recommendation input
    query = st.text_area(
        "What are you looking for?",
        placeholder="e.g., I need a charger cable under 300 rupees with good reviews and warranty",
        height=100
    )

    # Preferences
    with st.expander("🎛️ Preferences"):
        col1, col2 = st.columns(2)
        with col1:
            include_analysis = st.checkbox("Include Market Analysis", value=True)
        with col2:
            show_reasoning = st.checkbox("Show AI Reasoning", value=True)

    # Get recommendations
    if st.button("🎯 Get Recommendations", type="primary"):
        if not query:
            st.warning("Please describe what you're looking for")
        else:
            with st.spinner("Generating recommendations..."):
                try:
                    preferences = {
                        'include_analysis': include_analysis
                    }
                    result = system.get_recommendations(
                        query=query,
                        preferences=preferences
                    )
                    st.markdown("### Your Personalized Recommendations")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Failed to generate recommendations: {e}")


def tab_market_analysis(system):
    """Market Analysis Tab"""
    st.header("📊 Market Analysis")

    st.markdown("""
    Comprehensive market analysis using **Hybrid ML + GenAI**:
    - 🔹 Product Clustering (K-means)
    - 🎯 Quality Classification (Random Forest)
    - 💰 Price Segmentation
    - 😊 Sentiment Analysis
    - 🧠 LLM Interpretation
    """)

    st.markdown("---")

    # Run analysis button
    if st.button("📊 Run Market Analysis", type="primary"):
        with st.spinner("Running Hybrid ML + GenAI analysis... This may take a minute..."):
            try:
                result = system.analyze_market(use_cache=False)
                st.markdown(result)

                # Add download button
                st.download_button(
                    label="📥 Download Report",
                    data=result,
                    file_name="market_analysis_report.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    # Cached analysis
    if st.button("📋 View Cached Analysis"):
        with st.spinner("Loading cached analysis..."):
            try:
                result = system.analyze_market(use_cache=True)
                st.markdown(result)
            except Exception as e:
                st.error(f"Failed to load analysis: {e}")


def tab_qa(system):
    """Q&A Tab"""
    st.header("💬 Intelligent Q&A")

    st.markdown("""
    Ask any question about products, and our AI will:
    - 🔍 Search relevant information using Advanced RAG
    - 📊 Analyze data with ML models
    - 💡 Provide insights with LLM
    """)

    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about products..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = system.answer_question(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Example questions
    st.markdown("---")
    st.markdown("**💡 Example Questions:**")
    examples = [
        "What are the best value products?",
        "Show me affordable options with good ratings",
        "What's the market trend for pricing?",
        "Which products have the best reviews?",
        "Analyze the quality distribution"
    ]
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"Ask: {example}", key=f"ex_qa_{i}"):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


def main():
    """Main application"""

    # Initialize system
    system, success = initialize_system()

    if not success:
        st.error("Failed to initialize system. Please check configuration.")
        st.stop()

    # Render UI
    render_header()
    render_sidebar(system)

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Search",
        "🎯 Recommendations",
        "📊 Market Analysis",
        "💬 Q&A"
    ])

    with tab1:
        tab_search(system)

    with tab2:
        tab_recommendations(system)

    with tab3:
        tab_market_analysis(system)

    with tab4:
        tab_qa(system)


if __name__ == "__main__":
    main()
