import string
import numpy as np
import base64
import io
import joblib
import os
from nltk.tokenize import word_tokenize


def embeding(title):
    word2vec = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizers', 'baseline_vectorizer_v1.pkl'))
    tokens = word_tokenize(title.lower())
    title_embeddings_avg = []
    title_embedding = []
    for word in tokens:
        if word in word2vec.wv.key_to_index:
            word_embedding = word2vec.wv[word]
            title_embedding.append(word_embedding)
        if title_embedding:
            title_embedding_avg = np.mean(title_embedding, axis=0)
        else:
            title_embedding_avg = np.zeros(word2vec.vector_size)

    title_embeddings_avg = np.array(title_embedding_avg)

    return title_embeddings_avg

def title_preprocess(title):
    extracted_features = {}

    extracted_features['count_words'] = len(title.split())
    extracted_features['capital_ratio'] = sum(1 for c in title if c.isupper())
    extracted_features['letter_ratio'] = sum(1 for c in title if c.isalpha())
    extracted_features['punct_counts'] = sum(1 for c in title if c in string.punctuation)
    extracted_features['special_chars_ratio'] = sum(1 for c in title if c == ":" or c == "-")

    minmax_scale = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'scalers', 'baseline_title_feature_minmax_v1.pkl'))

    return minmax_scale.transform(np.array(list(extracted_features.values())).reshape(1, -1))

def image_preprocess(image):
    decoded_bytes = base64.b64decode(image)

    def bytes_to_np(bytes_data):
        byte_io = io.BytesIO(bytes_data)
        byte_io.seek(0)
        return np.load(byte_io)

    image_array = bytes_to_np(decoded_bytes)

    flattened_pixels = image_array.reshape(-1, 3)
    average_color = np.mean(flattened_pixels, axis=0)
    unique, counts = np.unique(flattened_pixels, return_counts=True, axis=0)
    most_frequent_index = np.argmax(counts)
    most_frequent_color = unique[most_frequent_index]

    return np.append(average_color, most_frequent_color)/255


def preprocess(image, title):
    return np.append(np.append(image_preprocess(image), title_preprocess(title)), embeding(title)).reshape(1, -1)
