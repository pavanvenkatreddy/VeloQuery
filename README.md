# VeloQuery

VeloQuery: A Vector Database Implementation with Text Search & Similarity
Overview
VeloQuery is a Python-based implementation of a Vector Database (VDB) that leverages pre-trained BERT embeddings for text similarity search. The system processes large text files, splits them into chunks, converts the text into vector embeddings using BERT, and allows efficient similarity searches. The VeloQuery project allows you to perform the following tasks:

* Convert text into vector representations using BERT embeddings.
* Store and index vectors for efficient retrieval.
* Find similar vectors to a query vector by comparing cosine similarity.
* Visualize the vectors and their relationships with the text chunks.

# Key Features

* Text-to-Vector Conversion: Uses BERT (Bidirectional Encoder Representations from Transformers) to generate vector embeddings from input text.
* Vector Storage: Supports storing vector data in a dictionary and performing similarity-based queries.
* Similarity Search: Implements a brute-force similarity search to retrieve the most similar vectors based on cosine similarity.
* Context Window: Allows breaking down large text files into smaller chunks and converting them into vectors, with a configurable context window.
* Visualization: Generates 2D plots of vector embeddings to visualize the relationships between different chunks of text and a query text.
* Scalability: Supports large text files by chunking and adding vector representations to the database.
