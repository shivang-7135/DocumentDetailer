from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

index_path = 'vector_indexNLP.faiss'
metadata_path = 'metadataNLP.json'

# Load FAISS index and metadata
index = faiss.read_index(index_path)
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

def convert_distance_to_similarity(distance):
# Assuming the distances are non-negative, we can use a simple conversion:
    return 1 / (1 + distance)*100

def query_index(query, model, index, metadata, top_k=5):
    query_embedding = model.encode(query).reshape(1,-1).astype('float32')
    D, I = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        doc_metadata = metadata[I[0, i]]
        similarity_score = convert_distance_to_similarity(D[0, i])
        result = {
            "filename": doc_metadata["filename"],
            "page_num": doc_metadata["page_num"],
            "standardized_text": doc_metadata["standardized_text"],
            "question_text":doc_metadata["question_text"],
            "answerable_text":doc_metadata["answerable_text"],
            "score":similarity_score
        }
        results.append(result)

    return results

query = "what is Rule-Based Machine Translation?"
results = query_index(query, model, index, metadata)



def create_answer_to_show(query, results):
    answer = f"Based on your query '{query}', the following relevant information was found:\n\n"
    for result in results:
        answer += "\n------------------------------------------------------------------------------------------------------------------\n"
        answer += f"Filename: {result['filename']}\n"
        answer += f"Page number: {result['page_num']}\n"
        answer += f"Related keywords:  {result['question_text'][:100]}...\n"
        if result['answerable_text']!="":
            answer += f"Answer: {result['answerable_text'][:500]}\n"
        answer += f"Relevancy Score: {result['score']}\n"
    answer += "\nFor more detailed information, please refer to the respective original texts.\n\n\n"
    return answer

answer = create_answer_to_show(query, results)

print(answer)
