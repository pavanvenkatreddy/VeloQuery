import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class VDB:
    def __init__(self, context_window_size=100):
        self.vector_data = {}  # A dictionary to store vectors
        self.vector_index = {}  # An indexing structure for retrieval
        self.context_window_size = context_window_size  # Context window size for chunking text
        
        # Initialize BERT Tokenizer and Model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()  # Set model to evaluation mode (no gradient calculation)

    def add_vector(self, vector_id, vector, text):
        """
        Add a vector to the store along with its associated text.
        """
        self.vector_data[vector_id] = {'vector': vector, 'text': text}
        self._update_index(vector_id, vector)
    
    def get_vector(self, vector_id):
        """
        Retrieve a vector and associated text from the store.
        """
        return self.vector_data.get(vector_id)
    
    def _update_index(self, vector_id, vector):
        """
        Update the index with the new vector.
        """
        for existing_id, existing_data in self.vector_data.items():
            existing_vector = existing_data['vector']
            similarity = np.dot(vector, existing_vector) / (np.linalg.norm(vector) * np.linalg.norm(existing_vector))
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][vector_id] = similarity
    
    def find_similar_vectors(self, query_vector, num_results=5):
        """
        Find similar vectors to the query vector using brute-force search.
        """
        results = []
        for vector_id, data in self.vector_data.items():
            vector = data['vector']
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity, data['text']))

        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        # Return the top N results
        return results[:num_results]
    
    def text_to_vector(self, text):
        """
        Convert text to a vector using BERT embeddings.
        """
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the embedding for the [CLS] token
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :].squeeze()  # First token [CLS] embedding
        
        return cls_embedding.numpy()  # Return the numpy array for the vector
    
    def process_large_text(self, large_text):
        """
        Process large text and add chunks to the database.
        Chunks will be split based on the context_window_size.
        """
        chunk_ids = []
        for i in range(0, len(large_text), self.context_window_size):
            chunk_text = large_text[i:i+self.context_window_size]
            chunk_id = len(chunk_ids)
            chunk_vector = self.text_to_vector(chunk_text)
            self.add_vector(chunk_id, chunk_vector, chunk_text)
            chunk_ids.append(chunk_id)

        return chunk_ids  # Return chunk IDs that were processed

    def plot_vectors(self, query_text, method="PCA"):
        """
        Plot the vectors in 2D, show chunk IDs, and highlight the query vector in red.
        """
        # Get the query vector
        query_vector = self.text_to_vector(query_text)

        # Extract the vectors and chunk IDs
        vectors = [data['vector'] for data in self.vector_data.values()]
        chunk_ids = [vector_id for vector_id in self.vector_data.keys()]
        
        # Include the query vector in the vectors list
        vectors.append(query_vector)
        chunk_ids.append("Query")  # Label for the query point
        
        # Reduce the dimensionality of vectors to 2D (using PCA or t-SNE)
        if method == "PCA":
            reduced_vectors = PCA(n_components=2).fit_transform(vectors)
        elif method == "t-SNE":
            reduced_vectors = TSNE(n_components=2, random_state=42).fit_transform(vectors)
        
        # Plot the reduced vectors
        plt.figure(figsize=(10, 8))
        
        # Plot all the chunk vectors (in blue)
        for i, (x, y) in enumerate(reduced_vectors[:-1]):
            plt.scatter(x, y, marker='o', color='b')
            plt.text(x, y, f"{chunk_ids[i]}", fontsize=9, ha='right', color='black')  # Label with chunk IDs
        
        # Plot the query vector (in red)
        x, y = reduced_vectors[-1]  # The last point is the query vector
        plt.scatter(x, y, marker='o', color='r', label="Query")
        plt.text(x, y, "Query", fontsize=12, ha='right', color='r', fontweight='bold')  # Label for the query
        
        plt.title("Vector Space of Text Chunks and Query")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(loc="upper right")
        plt.show()