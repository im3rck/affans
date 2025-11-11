# 🚀 Quick Start Guide

Get your Intelligent Retail Customer Support Agent running in 10 minutes!

## Prerequisites

- Python 3.9 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- 8GB RAM recommended

## 🎯 Quick Installation

### Option 1: Automated Setup (Recommended)

```bash
# 1. Navigate to project directory
cd retail_support_agent

# 2. Run automated setup
python setup.py
```

The script will:
- ✅ Check system requirements
- ✅ Install all dependencies
- ✅ Process the dataset
- ✅ Create vector database
- ✅ Verify installation

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...

# 4. Process data
python utils/data_preprocessor.py

# 5. Setup RAG system
python utils/rag_system.py
```

## 🔑 Configure API Key

1. Open `.env` file
2. Replace `your_openai_api_key_here` with your actual OpenAI API key:

```bash
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

## 🎬 Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🎮 First Steps

### 1. Initialize System
- Click **"Initialize System"** in the sidebar
- Wait for RAG system and agents to load (~30 seconds)

### 2. Try Sample Queries

**Product Search:**
```
What are the best USB cables under 500 rupees?
```

**Review Analysis:**
```
What do customers say about boAt cables?
```

**Product Comparison:**
```
Compare boAt and Ambrane cables
```

**Get Recommendations:**
```
I need a charging cable for my iPhone
```

## 📊 Explore Features

### Chat Support 💬
- Natural language queries
- Multi-turn conversations
- Context-aware responses

### Product Search 🔍
- Semantic search
- Filter by rating
- View product details

### Analytics Dashboard 📈
- System metrics
- Chat statistics
- RAGAS evaluation scores

## 🧪 Optional: Run Evaluation

```bash
# Run RAGAS evaluation
python utils/ragas_evaluation.py

# View results in Analytics dashboard
```

## 🔧 Optional: Fine-Tune Model

```bash
# Start OpenAI fine-tuning
python models/fine_tuning.py openai \
  --data data/processed/finetuning_data.jsonl

# Monitor job
python models/fine_tuning.py monitor --job-id <job_id>
```

## 🐛 Troubleshooting

### "Vector store not found"
```bash
python utils/data_preprocessor.py
python utils/rag_system.py
```

### "OpenAI API Error"
- Check API key in `.env`
- Verify account has credits
- Check internet connection

### "Module not found"
```bash
pip install -r requirements.txt --upgrade
```

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

## 📚 Learn More

- **README.md** - Complete documentation
- **Architecture details** - See README.md Architecture section
- **API Reference** - Check individual module docstrings

## 🎯 Quick Test Checklist

- [ ] System initialized successfully
- [ ] Can search for products
- [ ] Can ask questions in chat
- [ ] Analytics dashboard shows data
- [ ] No error messages

## 💡 Pro Tips

1. **First query is slower** - Embedding models need to warm up
2. **Use specific queries** - "USB cable under 500" vs "cheap cable"
3. **Try different query types** - Product search, reviews, comparisons
4. **Check analytics** - Monitor system performance
5. **Clear chat** - Use "Clear Chat" button for fresh start

## 🎓 Bootcamp Technologies Used

This project demonstrates:
- ✅ **Crew AI** - Multi-agent system
- ✅ **RAG** - ChromaDB + semantic search
- ✅ **Fine-tuning** - Custom model training
- ✅ **RAGAS** - Evaluation framework
- ✅ **Prompt Engineering** - 5 different strategies

## 🚀 Next Steps

After getting familiar with the basic features:

1. Explore different prompt strategies (Settings page)
2. Run RAGAS evaluation to see quality metrics
3. Fine-tune a model on the customer support data
4. Customize agents for your use case
5. Add more data sources

## 🤝 Need Help?

- Check **README.md** for detailed documentation
- Review **Troubleshooting** section above
- Check module docstrings for API details

---

**Ready to go!** 🚀 Start the app with `streamlit run app.py`
