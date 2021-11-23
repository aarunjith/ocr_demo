from celery import Celery
from process_inputs import process_pdf

app = Celery('OCR', broker="amqp://localhost:5672",
             backend="mongodb://localhost:27017/task_results")


@app.task(serializer='pickle')
def start_processing(pdf_bytes, ocr):
    _, result = process_pdf(pdf_bytes, ocr)
    return result
