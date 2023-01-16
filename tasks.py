from celery import Celery
import os
import sys

from numpy.lib.recfunctions import recursive_fill_fields

sys.path.append(os.getcwd())

RABBITMQ_URL = os.environ['RABBITMQ_URL']
MONGO_URL = os.environ['MONGO_URL']

app = Celery('OCR', broker=f"amqp://{RABBITMQ_URL}:5672",
             backend=f"mongodb://{MONGO_URL}:27017/task_results")


@app.task(bind=True)
def start_processing(self):
    from process_inputs import process_pdf
    from easyocr import Reader
    ocr = Reader(lang_list=['en'],
                 model_storage_directory='./weights', gpu=True)
    _, result = process_pdf('./data/temp.pdf', ocr)
    return result
