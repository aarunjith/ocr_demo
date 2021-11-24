from celery import Celery
import os
import sys

from numpy.lib.recfunctions import recursive_fill_fields

sys.path.append(os.getcwd())

app = Celery('OCR', broker="amqp://rabbitmq:5672",
             backend="mongodb://mongodb:27017/task_results")


@app.task(bind=True)
def start_processing(self):
    from process_inputs import process_pdf
    from easyocr import Reader
    ocr = Reader(lang_list=['en'],
                 model_storage_directory='./weights', gpu=True)
    _, result = process_pdf('./data/temp.pdf', ocr)
    return result
