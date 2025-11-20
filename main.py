
import argparse
from src.query_rag import query_rag, print_output

def main():
    parser = argparse.ArgumentParser(description="PDF Q&A with RAG")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("-s", "--similarity_threshold", type=float, default=0.25, help="Similarity score threshold")
    args = parser.parse_args()

    top_k = args.top_k
    similarity_threshold = args.similarity_threshold

    print("-" * 30)
    print("Welcome! [Type 'q' to quit.]")
    print("-" * 30, "\n")

    while True:
        user_input = input("Question: ")
        if user_input.lower() == 'q':
            break
        response_text, sources, results = query_rag(user_input, top_k=top_k, similarity_threshold=similarity_threshold)
        print_output(response_text, sources, DATA_PATH="src/data/")
        print("-" * 30)

if __name__ == "__main__":
    main()