import os
import dotenv
from sentence_transformers import SentenceTransformer, util

dotenv.load_dotenv()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#Fonction qui permet de trouver le sentence le plus proche de la query en utilisant cosine similarity

def find_closest(query, sentences):
    query_embeddings = model.encode(query)
    embeddings = model.encode(sentences)

    closest_index = 0
    closest_score = -1

    for index, vector in enumerate(embeddings):
        similarity_score = util.cos_sim(vector, query_embeddings).item()

        if similarity_score > closest_score:
            closest_score = similarity_score
            closest_index = index

    return sentences[closest_index]

def main():
    sentences = [
        "That is a happy person",
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]

    query = "I am great today"
    #Nous pouvons utliser notre fonction find_closest() ou bien utiliser directement semantic_search() de util
    # result = find_closest(query, sentences)
    # print(f"Closest sentence to '{query}': {result}")

    query_embeddings = model.encode(query)
    sentences_embeddings = model.encode(sentences)

    results = util.semantic_search(query_embeddings, sentences_embeddings, top_k=2)
    print(results)

if __name__ == "__main__":
    main()
