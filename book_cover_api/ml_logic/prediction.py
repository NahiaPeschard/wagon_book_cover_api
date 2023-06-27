import numpy as np

from book_cover_api.params import *

def prediction(X, model, categories=3):
    def top_k_argmax(array, k):
        indexes = np.argpartition(array, -k)[-k:]
        sorted_indexes = indexes[np.argsort(array[indexes])][::-1]
        return sorted_indexes

    y_pred = model.predict(X)[0]
    y_pred = [CATEGORY_DICT[pred] for pred in top_k_argmax(y_pred, categories)]

    return y_pred
