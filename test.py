from vecctor import VDB

vdb = VDB(context_window_size=500)  # Set context window size

# Read large text from file
with open('/Users/pavanvenkatreddy/Downloads/Algorithms.txt', 'r') as file:
    content = file.read()

# Process large text and add to the database
vdb.process_large_text(content)

# Query with text
query_text = "sales man problem, how to solve it!"
query_vector = vdb.text_to_vector(query_text)

# Find similar vectors
similar_vectors = vdb.find_similar_vectors(query_vector)

# Display similar vectors along with their associated text (first 100 chars)
print("Most similar vectors:")
for vector_id, similarity, text in similar_vectors:
    print(f"ID: {vector_id}, Similarity: {similarity:.4f}, Text: {text}")  # Displaying the first 100 chars

# Plot all vectors in 2D (PCA or t-SNE) and include the query vector as a red dot
vdb.plot_vectors(query_text, method="PCA")  # You can change to "t-SNE" for a different dimensionality reduction technique