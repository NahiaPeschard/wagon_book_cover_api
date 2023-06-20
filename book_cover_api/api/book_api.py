from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os
import joblib

from book_cover_api.params import *
from book_cover_api.ml_logic.preprocessor import *

app = FastAPI()
app.state.model = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'baseline_svc_v1.pkl'))

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.put("/predict")
def predict(body: dict):
    image_arr = body["image_arr"]
    title = body["title"]

    X_preproc = preprocess(image_arr, title)
    y_pred = app.state.model.predict(X_preproc)[0]

    return {"book_category": CATEGORY_DICT[y_pred]}
