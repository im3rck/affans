"""
Setup Script for Intelligent Retail System
Prepares all data and systems for first run
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("INTELLIGENT RETAIL SYSTEM - SETUP")
print("=" * 80)
print()

# Step 1: Check prerequisites
print("Step 1: Checking prerequisites...")
print("-" * 80)

# Check if data file exists
data_file = Path("./data/amazon.csv")
if not data_file.exists():
    print("❌ Error: amazon.csv not found in ./data/")
    print("   Please ensure amazon.csv is in the data directory")
    sys.exit(1)
else:
    print(f"✅ Found amazon.csv ({data_file.stat().st_size / (1024*1024):.1f} MB)")

# Check if required packages are installed
try:
    import pandas
    import numpy
    import chromadb
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    from sklearn.cluster import KMeans
    print("✅ All required packages installed")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Step 2: Preprocess data
print("\nStep 2: Preprocessing data...")
print("-" * 80)

from utils.data_preprocessor import AmazonDataPreprocessor

try:
    preprocessor = AmazonDataPreprocessor('./data/amazon.csv')
    df = preprocessor.load_data()
    rag_documents = preprocessor.create_rag_documents()
    training_data = preprocessor.create_finetuning_dataset()
    stats = preprocessor.save_processed_data(rag_documents, training_data, './data/processed')

    print("\n✅ Data preprocessing complete!")
    print(f"   Products: {stats['total_products']}")
    print(f"   Training examples: {stats['total_training_examples']}")
    print(f"   Categories: {stats['categories']}")
    print(f"   Average rating: {stats['avg_rating']:.2f}/5")
except Exception as e:
    print(f"❌ Data preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Create vectorstore
print("\nStep 3: Creating vectorstore...")
print("-" * 80)

from utils.rag_system import RAGSystem

try:
    # Use FREE local embeddings
    rag = RAGSystem(
        persist_directory="./vectorstore/chroma_db",
        use_openai=False
    )

    # Load processed documents
    import json
    with open('./data/processed/rag_documents.json', 'r') as f:
        documents = json.load(f)

    # Create vectorstore
    texts = [doc['text'] for doc in documents]
    metadatas = [doc['metadata'] for doc in documents]

    rag.create_vectorstore(texts, metadatas)
    print(f"\n✅ Vectorstore created with {len(documents)} documents!")

except Exception as e:
    print(f"❌ Vectorstore creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Verify setup
print("\nStep 4: Verifying setup...")
print("-" * 80)

errors = []

# Check processed data
if not Path("./data/processed/processed_products.csv").exists():
    errors.append("processed_products.csv not found")
else:
    print("✅ processed_products.csv created")

if not Path("./data/processed/rag_documents.json").exists():
    errors.append("rag_documents.json not found")
else:
    print("✅ rag_documents.json created")

# Check vectorstore
if not Path("./vectorstore/chroma_db").exists():
    errors.append("vectorstore not found")
else:
    print("✅ Vectorstore created")

# Try loading the intelligent system
try:
    from utils.intelligent_system import IntelligentRetailSystem
    print("✅ Can import IntelligentRetailSystem")
except Exception as e:
    errors.append(f"System import failed: {e}")

if errors:
    print("\n⚠️ Setup completed with warnings:")
    for error in errors:
        print(f"   - {error}")
    print("\nYou may still be able to use some features.")

# Success!
print("\n" + "=" * 80)
print("✅ SETUP COMPLETE!")
print("=" * 80)
print()
print("Your Intelligent Retail System is ready to use!")
print()
print("Next steps:")
print()
print("1. Run the web interface:")
print("   streamlit run app_new.py")
print()
print("2. Run the demo (requires OpenAI API key):")
print("   python demo.py")
print()
print("3. Run FREE demo (no API key required):")
print("   python demo_free.py")
print()
print("4. Use Python API:")
print("   from utils.intelligent_system import IntelligentRetailSystem")
print("   system = IntelligentRetailSystem(use_openai_embeddings=False)")
print()
print("Note: Some features require OpenAI API key (set in .env file)")
print("      But basic search works with FREE local embeddings!")
print("=" * 80)
