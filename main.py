import os

# Ensure Transformers doesn't try to import TensorFlow/Keras when loading sentence-transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from src.prepare_data import prepare_car_reviews_data
from src.chromadb_manager import (
    create_chromadb_collection,
    insert_reviews_to_chromadb,
    query_reviews_from_chromadb,
)
from src.ai_model import ReviewAnalyzer

DATA_PATH = "D:\\market_car_analysis\\data\\archive"
COLLECTION_NAME = "car_reviews"

if __name__ == "__main__":
    # 1) Prepare data
    car_reviews = prepare_car_reviews_data(DATA_PATH)

    # 2) Create/load Chroma collection and insert
    collection = create_chromadb_collection(name=COLLECTION_NAME)
    insert_reviews_to_chromadb(collection, car_reviews)

    # 3) Query
    user_question = "What do 2017 models owners like most about their cars?"
    results = query_reviews_from_chromadb(collection, user_question, n_results=5)

    # 4) Analyze with local Ollama DeepSeek R1 via LangChain
    analyzer = ReviewAnalyzer()
    answer = analyzer.analyze_with_results(results, user_question, max_items=5)

    print("\n=== DeepSeek R1 Answer ===\n")
    print(answer)
