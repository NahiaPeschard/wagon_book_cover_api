import string
import numpy as np
import os
import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import joblib
import math
from PIL import ImageStat, Image
import cv2
from colorthief import ColorThief
import colorsys
from io import BytesIO
from gensim.models import Word2Vec
import gensim.downloader
import concurrent.futures

from langdetect import detect
from deep_translator import GoogleTranslator
from tensorflow.keras.preprocessing.sequence import pad_sequences

from google.cloud import vision

from spellchecker import SpellChecker

nltk.download('punkt')
nltk.download('stopwords')

def get_title(contents):
    client = vision.ImageAnnotatorClient.from_service_account_json(os.path.join(os.path.dirname(__file__), "..", '..', 'wagon-bootcamp-384313-f2ce2b3b3bf6.json'))

    image = vision.Image(content=contents)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        title = texts[0].description.replace('\n', ' ')
    else:
        title = ''

    return title.strip()

def get_labels(contents):
    client = vision.ImageAnnotatorClient.from_service_account_json(os.path.join(os.path.dirname(__file__), "..", '..', 'wagon-bootcamp-384313-f2ce2b3b3bf6.json'))

    image = vision.Image(content=contents)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    output_labels = [label.description.lower() for label in labels]
    return output_labels


def get_special_title_features(title):
    has_special = int(':' in title or '-' in title)
    has_comma = int(',' in title)
    return np.array([has_special, has_comma])


def translate(title):
    try:
        result_lang = detect(title.lower())
        if result_lang != "en":
            translator = GoogleTranslator(source='auto', target='en')
            title = translator.translate(title)
        return title
    except:
        return title


def spell_check(title):
    title = title.lower()
    title = "".join([i for i in title if i not in string.punctuation])
    title = "".join([i for i in title if i not in string.digits])
    title = re.split('\W+', title)

    spell = SpellChecker()
    misspelled = spell.unknown(title)

    fixed_words = []
    for word in misspelled:
        candidates = spell.candidates(word)
        if candidates is not None and len(candidates) <= 5:
            correction = spell.correction(word)
            title = [correction if token == word else token for token in title]
            fixed_words.append(word)

    misspelled.difference_update(fixed_words)

    title_copy = title.copy()
    for i, word in enumerate(title_copy[:-1]):
        if word in misspelled and title_copy[i+1] in misspelled:
            combo_word = word + title_copy[i+1]
            candidates = spell.candidates(combo_word)
            if candidates is not None and len(candidates) <= 3:
                title[i] = spell.correction(combo_word)
                title[i+1] = ""

    title = [word for word in title if word != ""]
    return title


def clean(title):
    porter_stemmer = PorterStemmer()
    title = [porter_stemmer.stem(word) for word in title]
    stops = set(stopwords.words('english'))
    title = [word for word in title if word not in stops]
    title = [word for word in title if len(word) > 2]
    return title

def embed(title, labels):
    word2vec = gensim.downloader.load("glove-wiki-gigaword-50")
    embedded_words = [word2vec[word] for list in (title, labels) for word in list if word in word2vec]
    padded_feature = pad_sequences([embedded_words], dtype='float32', padding='post', value=0, maxlen=100)
    return padded_feature

def get_brightness(contents):
    img_file = BytesIO(contents)
    image = Image.open(img_file)
    stat = ImageStat.Stat(image)
    r,g,b = stat.rms
    return int(math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)))

def detect_face(contents):
    image_array = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(image_array, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(img)
    if len(face) > 0:
        return 1
    else:
        return 0

def count_colors(contents):
    image_array = np.fromstring(contents, np.uint8)
    img_temp = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    N = img_temp.shape[0] * img_temp.shape[1]
    X = img_temp.reshape((N, 3))
    color_count = len(np.unique(X, axis=0))
    return color_count

def main_color_hls(contents):
    img_file = BytesIO(contents)
    color_thief = ColorThief(img_file)
    color = color_thief.get_color()
    hls = colorsys.rgb_to_hls(color[0]/255, color[1]/255, color[2]/255)
    return hls

def preprocess_image(contents):
    brightness = get_brightness(contents)
    no_colors = count_colors(contents)
    mm_scaler = joblib.load(os.path.join(os.path.dirname(__file__), "..", 'models', 'scalers', 'mm_scaler.pkl'))
    scaled = mm_scaler.transform(np.array([brightness, no_colors]).reshape(1, -1))
    face = detect_face(contents)
    hue, lightness, saturation = main_color_hls(contents)
    image_features = np.array([face, hue, lightness, saturation])
    return np.hstack([scaled.flatten(), image_features])


def preprocess(contents):
    title = get_title(contents)
    special_features = get_special_title_features(title)
    X_features = np.hstack([preprocess_image(contents), special_features])
    title = translate(title)
    title = spell_check(title)
    title = clean(title)
    labels = get_labels(contents)
    X_text = embed(title, labels)
    return X_text, X_features

def parallel_preprocess(contents):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the functions to the executor for parallel execution
        title_future = executor.submit(get_title, contents)
        special_features_future = executor.submit(get_special_title_features, title_future.result())
        image_features_future = executor.submit(preprocess_image, contents)
        labels_future = executor.submit(get_labels, contents)

        # Wait for the results
        title = title_future.result()
        special_features = special_features_future.result()
        image_features = image_features_future.result()
        labels = labels_future.result()

    X_features = np.hstack([image_features, special_features])

    # Preprocess the title
    title = translate(title)
    title = spell_check(title)
    title = clean(title)

    # Embed the preprocessed title and labels
    X_text = embed(title, labels)

    return X_text, X_features
