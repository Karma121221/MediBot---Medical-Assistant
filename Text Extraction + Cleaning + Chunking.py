import os
import fitz  # PyMuPDF
import nltk
import re

# Download NLTK tokenizer
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Set your PDF folder path
PDF_FOLDER = "/Users/rajeevranjanpratapsingh/PycharmProjects/healthcare/medical data"  # <-- Change this!

# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Optional: Clean text (remove headers/footers/symbols and normalize)
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)  # Remove newlines
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    text = re.sub(r'\[\d+\]', '', text)  # Remove reference [1], [2], etc.
    text = re.sub(r'\s{2,}', ' ', text)  # Remove extra spaces
    return text.lower().strip()

# Split text into smaller chunks
def chunk_text(text, max_tokens=500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Main script to process all PDFs in a folder
def process_all_pdfs(folder_path):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            raw_text = extract_text_from_pdf(file_path)
            cleaned_text = clean_text(raw_text)
            chunks = chunk_text(cleaned_text)
            all_chunks.extend(chunks)
    print(f"✅ Finished processing. Total chunks created: {len(all_chunks)}")
    return all_chunks

# Run the script
if __name__ == "__main__":
    chunks = process_all_pdfs(PDF_FOLDER)

    # Optional: Save all chunks to a file
    with open("new_chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"[Chunk {i+1}]\n{chunk}\n\n")
