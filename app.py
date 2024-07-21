from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import re
import shutil

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return render_template('index.html')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = SentenceTransformer('all-MiniLM-L6-v2')

index_path = 'vector_index1.faiss'
metadata_path = 'metadata1.json'

embeddings = []
metadata = []

def tensor_to_string(tensor):
    return tensor.numpy().decode("utf-8")

def extract_text_from_pdf_with_page_numbers(pdf_path):
    doc = fitz.open(pdf_path)
    text_pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        text_pages.append((page_num + 1, text))
    return text_pages

def custom_standardization(input_data):
    index_pattern = re.compile(r'\.{3,}')
    if bool(index_pattern.search(input_data.numpy().decode('utf-8'))):
        return ""
    stripped_urls = tf.strings.regex_replace(input_data, r"https?://\S+|www\.\S+", "")
    stripped_emails = tf.strings.regex_replace(stripped_urls, r"\S+@\S+", "")
    stripped_brackets = tf.strings.regex_replace(stripped_emails, r"<.*?>", "")
    stripped_square_brackets = tf.strings.regex_replace(stripped_brackets, r"\[|\]", "")
    stripped_digits = tf.strings.regex_replace(stripped_square_brackets, r"\w*\d\w*", "")
    stripped_non_alpha = tf.strings.regex_replace(stripped_digits, r"[^a-zA-Z\s]", "")
    standardized_text = tf.strings.regex_replace(stripped_non_alpha, r"\s+", " ")
    return standardized_text.numpy().decode('utf-8')

def split_into_paragraphs(text):
    pattern = r'(?<=\n)(?=\d+)'
    paragraphs = re.split(pattern, text)
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    return paragraphs

def text_to_vectors(paragraphs):
    vectors = model.encode(paragraphs)
    return vectors

def split_into_qa(text):
    # Define the regex pattern to capture the question and answer in one line
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

def store_vectors(paragraphs, vectors, metadata, filename, page_num):
    for i, (paragraph, vector) in enumerate(zip(paragraphs, vectors)):
        original_text = paragraph
        question, answer = split_into_qa(original_text)
        original_text = paragraph[:500]
        standardized_text = custom_standardization(tf.constant(paragraph))
        vector = model.encode(standardized_text).tolist()
        metadata.append({
            "index": f'paragraph-{i}',
            "filename": filename,
            "page_num": page_num,
            "standardized_text": standardized_text,
            "question_text": question,
            "answerable_text": answer
        })
        embeddings.append(vector)

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # filename = secure_filename(file.filename)
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # file.save(file_path)

        filename = secure_filename(file.filename)
        
        # Delete the uploads folder and its contents
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
        
        # Recreate the uploads folder
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            os.remove('metadata1.json')
            os.remove('vector_index1.faiss')
        except OSError as e:
            print(f"Error: {e.strerror}")
        process_pdf(file_path, filename)
        print(file_path+filename)
        return jsonify({'success': 'File uploaded and processed successfully'})

def process_pdf(file_path, filename):
    text_pages = extract_text_from_pdf_with_page_numbers(file_path)
    for page_num, text in text_pages:
        paragraphs = split_into_paragraphs(text)
        vectors = text_to_vectors(paragraphs)
        store_vectors(paragraphs, vectors, metadata, filename, page_num)
    save_index_and_metadata()

def save_index_and_metadata():
    embeddings_array = np.array(embeddings, dtype='float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    batch_size = 1000
    for i in range(0, len(embeddings), batch_size):
        batch_embeddings = embeddings_array[i:i+batch_size]
        index.add(batch_embeddings)
    faiss.write_index(index, index_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)



# Load FAISS index and metadata


def convert_distance_to_similarity(distance):
    # Assuming the distances are non-negative, we can use a simple conversion:
    return 1 / (1 + distance) * 100

def query_index(query, model, index, metadata, top_k=5):
    query_embedding = model.encode(query).reshape(1, -1).astype('float32')
    D, I = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        doc_metadata = metadata[I[0, i]]
        similarity_score = convert_distance_to_similarity(D[0, i])
        result = {
            "filename": doc_metadata["filename"],
            "page_num": doc_metadata["page_num"],
            "standardized_text": doc_metadata["standardized_text"],
            "question_text": doc_metadata["question_text"],
            "answerable_text": doc_metadata["answerable_text"],
            "score": similarity_score
        }
        results.append(result)

    return results

def create_answer_to_show(query, results):
    answer = f"Based on your query '{query}', the following relevant information was found:\n\n"
    for result in results:
        answer += "\n------------------------------------------------------------------------------------------------------------------\n"
        answer += f"Filename: {result['filename']}\n"
        answer += f"Page number: {result['page_num']}\n"
        answer += f"Related keywords: {result['question_text']}...\n"
        if result['answerable_text'] != "":
            answer += f"Answer: {result['answerable_text'][:500]}\n"
        answer += f"Relevancy Score: {result['score']}\n"
    answer += "\nFor more detailed information, please refer to the respective original texts.\n\n\n"
    return answer

@app.route('/api/query', methods=['POST'])
def query_endpoint():
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 5)
    index = faiss.read_index(index_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    results = query_index(query, model, index, metadata, top_k)
    formatted_answer = create_answer_to_show(query, results)
    
    return jsonify({'answer': formatted_answer})



if __name__ == '__main__':
    app.run(debug=True)
