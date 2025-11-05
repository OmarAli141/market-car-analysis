# Car Market Analysis - RAG System

A Retrieval-Augmented Generation (RAG) system for analyzing car reviews using ChromaDB vector storage and Ollama DeepSeek R1.

## Features

- **Data Processing**: Processes car review CSV files with Polars
- **Vector Storage**: Stores reviews in ChromaDB with sentence transformer embeddings
- **Semantic Search**: Queries similar reviews based on user questions
- **AI Analysis**: Generates answers using local Ollama DeepSeek R1 model

## Setup

1. **Install Python dependencies:**
```bash
pip install polars chromadb sentence-transformers ollama langchain-community langchain-core tf-keras
```

2. **Install and start Ollama:**
   - Download from [ollama.com](https://ollama.com/download)
   - Pull the model: `ollama pull deepseek-r1:1.5b`

3. **Ensure data files are in `data/archive/` directory**

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Load and process car review data
2. Create/update ChromaDB collection with embeddings
3. Query similar reviews for the question
4. Generate an AI-powered answer using DeepSeek R1

## Project Structure

```
market_car_analysis/
├── main.py                 # Main entry point
├── src/
│   ├── prepare_data.py     # Data preparation and ETL
│   ├── chromadb_manager.py # ChromaDB operations
│   └── ai_model.py         # Ollama DeepSeek R1 integration
└── data/
    └── archive/            # Car review CSV files
```

## Configuration

- Edit `DATA_PATH` in `main.py` to change data directory
- Modify `user_question` in `main.py` to ask different questions
- Adjust `vehicle_years` in `prepare_data.py` to filter by year

