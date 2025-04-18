from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load your chunks from the previous script
def load_chunks(file_path="/Users/rajeevranjanpratapsingh/PycharmProjects/healthcare/new_chunks.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = content.split("\n\n")
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return cleaned_chunks

# Initialize model and generate embeddings
def embed_chunks(chunks):
    print("🔍 Generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# Store embeddings in FAISS
def store_in_faiss(embeddings, chunks):
    print("📦 Saving to FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Save index and chunks for retrieval
    faiss.write_index(index, "new_faiss_index.idx")
    with open("new_chunk_texts.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Stored {len(chunks)} chunks in FAISS.")

# Execute the steps
if __name__ == "__main__":
    chunks = load_chunks("all_chunks.txt")
    embeddings = embed_chunks(chunks)
    store_in_faiss(embeddings, chunks)
