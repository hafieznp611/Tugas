import math
import os
import sys
import nltk

from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download data nltk jika belum ada
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# =========================================
# PREPROCESSING
# =========================================
def preprocess(text):
    tokens = word_tokenize(text.lower())

    result = []

    for token in tokens:
        if token.isalnum() and token not in stop_words:
            stemmed = stemmer.stem(token)
            result.append(stemmed)

    return result

# =========================================
# MEMBACA FILE DOKUMEN
# =========================================
def load_documents(base_file):
    docs = {}

    with open(base_file, 'r') as f:
        filenames = [line.strip() for line in f]

    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as file:
            docs[filename] = file.read()

    return docs

# =========================================
# TF
# =========================================
def compute_tf(tokens):
    freq = Counter(tokens)
    tf = {}

    for term, count in freq.items():
        tf[term] = 1 + math.log10(count)

    return tf

# =========================================
# IDF
# =========================================
def compute_idf(processed_docs):
    N = len(processed_docs)

    df = defaultdict(int)

    for tokens in processed_docs.values():
        unique_terms = set(tokens)

        for term in unique_terms:
            df[term] += 1

    idf = {}

    for term, n in df.items():
        idf[term] = math.log10(N / n)

    return idf

# =========================================
# TF-IDF
# =========================================
def compute_tfidf(tf, idf):
    tfidf = {}

    for term, value in tf.items():
        tfidf[term] = value * idf.get(term, 0)

    return tfidf

# =========================================
# COSINE SIMILARITY
# =========================================
def cosine_similarity(vec1, vec2):
    dot_product = 0

    for term in vec1:
        dot_product += vec1.get(term, 0) * vec2.get(term, 0)

    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (norm1 * norm2)

# =========================================
# MAIN PROGRAM
# =========================================
def main():

    if len(sys.argv) != 3:
        print("Usage: python vsm.py base.txt query.txt")
        return

    base_file = sys.argv[1]
    query_file = sys.argv[2]

    # Load dokumen
    documents = load_documents(base_file)

    # Preprocess dokumen
    processed_docs = {}

    for filename, text in documents.items():
        processed_docs[filename] = preprocess(text)

    # Hitung TF
    tf_docs = {}

    for filename, tokens in processed_docs.items():
        tf_docs[filename] = compute_tf(tokens)

    # Hitung IDF
    idf = compute_idf(processed_docs)

    # Hitung TF-IDF dokumen
    tfidf_docs = {}

    for filename, tf in tf_docs.items():
        tfidf_docs[filename] = compute_tfidf(tf, idf)

    # =====================================
    # QUERY
    # =====================================
    with open(query_file, 'r', encoding='utf-8') as f:
        query = f.read()

    query_tokens = preprocess(query)

    tf_query = compute_tf(query_tokens)

    tfidf_query = compute_tfidf(tf_query, idf)

    # =====================================
    # SIMILARITY
    # =====================================
    scores = []

    for filename, vector in tfidf_docs.items():
        score = cosine_similarity(vector, tfidf_query)

        if score > 0:
            scores.append((filename, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    # =====================================
    # OUTPUT response.txt
    # =====================================
    with open("response.txt", "w") as f:
        f.write(str(len(scores)) + "\n")

        for filename, score in scores:
            f.write(f"{filename} {score:.4f}\n")

    # =====================================
    # OUTPUT weights.txt
    # =====================================
    with open("weights.txt", "w") as f:

        for filename, weights in tfidf_docs.items():

            f.write(f"{filename}: ")

            for term, weight in weights.items():
                f.write(f"{term},{weight:.4f} ")

            f.write("\n")

    # =====================================
    # OUTPUT index.txt
    # =====================================
    inverted_index = defaultdict(list)

    for filename, weights in tfidf_docs.items():

        for term, weight in weights.items():
            inverted_index[term].append((filename, weight))

    with open("index.txt", "w") as f:

        for term, docs in inverted_index.items():

            f.write(f"{term}: ")

            for doc, weight in docs:
                f.write(f"{doc},{weight:.4f} ")

            f.write("\n")

    print("Program selesai.")
    print("Output:")
    print("- index.txt")
    print("- weights.txt")
    print("- response.txt")

# =========================================

if __name__ == "__main__":
    main()