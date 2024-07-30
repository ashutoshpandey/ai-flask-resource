import json
import numpy as np
import pandas as pd
from utils.pinecone_util import PineconeUtil
from utils.embedding_util import EmbeddingUtil

# Load data and create embeddings
def load_csv():
    pinecone_util = PineconeUtil(True)
    embedding_util = EmbeddingUtil()

    print('Loading csv...')
    data = load_csv('data.csv')
    embeddings = embedding_util.create_embeddings(data) 
    pinecone_util.store_embeddings(embeddings)


# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Process the query
def query_database(query, original_data):
    pinecone_util = PineconeUtil()
    embedding_util = EmbeddingUtil()

    query_embedding = embedding_util.generate_embedding(query)
    result = pinecone_util.process_query(query_embedding)

    return process_result(result, original_data, query_embedding)


# Process the response
def process_result(result, original_data, query_embedding, similarity_threshold=0.2):
    embedding_util = EmbeddingUtil()

    json_results = []
    for match in result['matches']:
        user_data = original_data.iloc[int(match['id'])][['name', 'email', 'designation', 'skills']]

        # next 3 lines are for debugging
        user_embedding = embedding_util.generate_embedding(f"{user_data['name']} {user_data['email']} {user_data['designation']} {' '.join(user_data['skills'])}")
        similarity = cosine_similarity(query_embedding, user_embedding)
        print(f"Similarity with '{user_data['name']}': {similarity}")
        
        if similarity >= similarity_threshold:
            json_results.append({
                'name': user_data['name'],
                'email': user_data['email'],
                'designation': user_data['designation'],
                'skills': user_data['skills']
            })

    # Convert list of dictionaries to JSON string
    json_string = json.dumps(json_results, indent=4)
    print(json_string)
    return json_results
