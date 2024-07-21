import os
import fitz  # PyMuPDF
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import re

# This folder should contain all the pdf files which we need to work on . Below given is just an example
pdf_folder = '/Users/shivangsinha/Downloads/personalProject'
pdf_text_data = {}
embeddings = []
metadata = []

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
#model = SentenceTransformer('paraphrase-MiniLM-L6-v2') - Also tried with other model but seems the current one is working better.

# converting tensor to string so that to store it in json format.
def tensor_to_string(tensor):
    return tensor.numpy().decode("utf-8")  # Assuming utf-8 encoding

# extract text based on page number so that it is more relevant for search. 
def extract_text_from_pdf_with_page_numbers(pdf_path):
    doc = fitz.open(pdf_path)
    text_pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        text_pages.append((page_num + 1, text))  # Page numbers are 1-based in fitz

    return text_pages

# Making sure inout data is not coming from table of content part and also preprocess all the text which are irrevant for the search.
def custom_standardization(input_data):
    
    # If index pattern is seems to be part of table of content then simply ignore it.
    index_pattern = re.compile(r'\.{3,}')
    if bool(index_pattern.search(input_data.numpy().decode('utf-8'))):
        return ""

    # Remove URLs
    stripped_urls = tf.strings.regex_replace(input_data, r"https?://\S+|www\.\S+", "")

    # Remove email addresses
    stripped_emails = tf.strings.regex_replace(stripped_urls, r"\S+@\S+", "")

    # Remove text in angular brackets (usually HTML tags)
    stripped_brackets = tf.strings.regex_replace(stripped_emails, r"<.*?>", "")

    # Remove any square brackets and leave the text within square brackets
    stripped_square_brackets = tf.strings.regex_replace(stripped_brackets, r"\[|\]", "")

    # Remove alphanumeric characters with digits
    stripped_digits = tf.strings.regex_replace(stripped_square_brackets, r"\w*\d\w*", "")

    # Remove non-alphabet characters
    stripped_non_alpha = tf.strings.regex_replace(stripped_digits, r"[^a-zA-Z\s]", "")

    # Replace multiple whitespaces with a single whitespace
    standardized_text = tf.strings.regex_replace(stripped_non_alpha, r"\s+", " ")

    return standardized_text.numpy().decode('utf-8')


# For the time being I am using the pattern of question and answer. I am splitting up text into paragraphs which ends with ? mark
def split_into_paragraphs(text):
    pattern = r'(?<=\n)(?=\d+\.)'
    
    # Split text using the pattern
    paragraphs = re.split(pattern, text)
    
    # Remove leading/trailing whitespace from each paragraph and filter out empty paragraphs
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    
    return paragraphs


# This part is for storing the vector of a paragraph in a required format
def text_to_vectors(paragraphs):
    vectors = model.encode(paragraphs)
    return vectors

# This split is used to Answer the query or simply show the relevant text from the book.
def split_into_qa(text):
    # Find the last occurrence of a question mark
    index_pattern = re.compile(r'\.{3,}')
    # Split the text at each question mark followed by a newline or space
    match = re.search(r'(.*\?.*?)\n', text, re.DOTALL)
    
    # If a match is found, split the text accordingly
    if match:
        question = match.group(1).strip()  # The part before the last question mark
        answer = text[match.end():].strip()  # The part after the last question mark
        
        # Filter out index-like entries in both question and answer
        if index_pattern.search(question):
            question = ""  # Ignore this as it looks like an index entry
        if index_pattern.search(answer):
            answer = ""  # Ignore this as it looks like an index entry
    else:
        question = text.strip()  # No question mark found, consider the entire text as the question
        answer = ""  # No answer part
    
    return question, answer

# storing vector to use it later while querying
def store_vectors(paragraphs, vectors, metadata, filename, page_num):
    for i, (paragraph, vector) in enumerate(zip(paragraphs, vectors)):
        original_text = paragraph
        question,answer = split_into_qa(original_text)
        original_text = paragraph[:500]  # Store the first 500 characters of the original text
        standardized_text = custom_standardization(tf.constant(paragraph))
        vector = model.encode(standardized_text).tolist()  # Recompute vector for standardized text
        metadata.append({
            "index": f'paragraph-{i}',
            "filename": filename,
            "page_num": page_num,
            "standardized_text": standardized_text,
            "question_text":question,
            "answerable_text":answer
        })
        embeddings.append(vector)

for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, filename)
        text_pages = extract_text_from_pdf_with_page_numbers(pdf_path)
        for page_num, text in text_pages:
            paragraphs = split_into_paragraphs(text)
            vectors = text_to_vectors(paragraphs)
            store_vectors(paragraphs, vectors, metadata, filename, page_num)
        pdf_text_data[filename] = text_pages

# Save FAISS index and metadata to JSON
index_path = 'vector_indexNLP.faiss'
metadata_path = 'metadataNLP.json'

# Convert embeddings to numpy array for FAISS
embeddings_array = np.array(embeddings, dtype='float32')

# Initialize FAISS index
dimension = embeddings_array.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)

# Add embeddings in batches to avoid memory issues. I faced some issue while adding index
batch_size = 1000  # Adjust batch size based on available memory
for i in range(0, len(embeddings), batch_size):
    batch_embeddings = embeddings_array[i:i+batch_size]
    index.add(batch_embeddings)

# Save the FAISS index
faiss.write_index(index, index_path)

# Save metadata
with open(metadata_path, 'w') as f:
    json.dump(metadata, f)

print(f"FAISS index saved to: {index_path}")
print(f"Metadata saved to: {metadata_path}")
