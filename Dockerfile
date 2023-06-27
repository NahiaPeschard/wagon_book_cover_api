FROM python:3.10.6-buster

COPY book_cover_api/api /book_cover_api/api
COPY book_cover_api/ml_logic /book_cover_api/ml_logic
COPY book_cover_api/models /book_cover_api/models
COPY book_cover_api/params.py /book_cover_api/params.py
COPY requirements_prod.txt requirements.txt
copy wagon-bootcamp-384313-f2ce2b3b3bf6.json wagon-bootcamp-384313-f2ce2b3b3bf6.json
COPY setup.py setup.py

RUN pip install --upgrade pip
RUN pip install .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD uvicorn book_cover_api.api.book_api:app --host 0.0.0.0 --port $PORT
