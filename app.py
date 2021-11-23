from typing import Optional
from uuid import uuid1
from xml.etree.ElementTree import dump
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from easyocr import Reader
from src.image_utils import convert_bbs
from src.image_registration import dump_template, convert_to_pascal
import numpy as np
from loguru import logger
from starlette.responses import FileResponse, JSONResponse
from pdf2image import convert_from_bytes
from process_inputs import process_pdf
from uuid import uuid1
from tasks import start_processing
from celery.result import AsyncResult
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
# Database Name
db = client["task_results"]
# Collection Name
coll = db["celery_taskmeta"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

ocr = Reader(lang_list=['en'], model_storage_directory='./weights', gpu=True)


@app.post('/transcribe')
async def transcribe_crop(img: UploadFile = File(...)):
    img = np.array(Image.open(BytesIO(img.file.read())))
    if img.max() == 1:
        img *= 255
    img = img.astype('uint8')
    output_dict = ocr.recognize(img, output_format='dict')
    return output_dict


@app.post('/get_text')
async def transcribe_crop(img: UploadFile = File(...)):
    img = np.array(Image.open(BytesIO(img.file.read())))
    if img.max() == 1:
        img *= 255
    img = img.astype('uint8')
    outputs = ocr.readtext(img, output_format='dict')
    outputs = [{'boxes': convert_bbs(output['boxes']),
                'text': output['text'],
                'confidence': output['confident']} for output in outputs]
    return outputs


@app.post('/pdf_template')
async def process_pdf_from_template(img: UploadFile = File(...)):
    registered_img, result = process_pdf(img.file.read(), ocr)
    image = Image.fromarray(registered_img)
    impath = f'{str(uuid1())}.jpg'
    image.save('data/' + impath)
    result['image'] = impath
    return result


@app.post('/batch_process')
async def process_pdf_async(img: UploadFile = File(...)):
    try:
        with open('./data/temp.pdf', 'wb') as file:
            file.write(img.file.read())
        result = start_processing.delay()
        return {"status": result.state, 'id': result.id, 'error': ''}
    except Exception as e:
        return {"status": 'FAILURE', 'error': e}


@app.post('/check_progress/{task_id}')
async def check_async_progress(task_id: str):
    try:
        result = AsyncResult(task_id)
        if result.ready():
            data = coll.find({'_id': task_id})[0]
            return {'status': 'SUCEESS', 'data': data['result']}
        else:
            return {"status": result.state, "error": ''}
    except Exception as e:
        data = coll.find({'_id': task_id})[0]
        if data:
            return {'status': 'SUCEESS', 'data': data['result']}
        return {'status': 'Task ID invalid', 'error': e}


@app.post('/add_template')
async def add_pdf_template(img: UploadFile = File(...)):
    try:
        logger.info('Processing PDF ....')
        ims = convert_from_bytes(img.file.read())
        first_page = np.array(ims[0])
        filename = f'template_{str(uuid1())}'
        logger.info('Dumping Template image and keypoints')
        dump_template(first_page, filename)
        logger.info('DONE!')
        result = {'status': 'SUCCESS', 'fileid': filename, 'error': ''}
    except Exception as e:
        result = {'status': 'FAILURE', 'error': e}
    return result


@app.post('/add_labels')
async def add_labels(labeldata: Request):
    try:
        logger.info('Request recieved')
        labeldata = await labeldata.json()
        filename = labeldata['filename']
        annotations = labeldata['Annotation']
        convert_to_pascal(annotations, filename)
        result = {'status': 'SUCCESS', 'error': ''}
    except Exception as e:
        result = {'status': 'FAILURE', 'error': e}
    logger.info(result)
    return result


@app.get('/image/{impath}')
async def show_image(impath: str, template: Optional[str] = None):
    if not template:
        return FileResponse(f'data/{impath}')
    elif template == 'yes':
        return FileResponse(f'templates/{impath}.jpg')


@app.get("/")
async def main():
    content = """
                <body>
                <h2>Single Crop</h2>
                <form action="/transcribe/" enctype="multipart/form-data" method="post">
                <input name="img" type="file">
                <input type="submit">
                </form>

                <h2>Full Image</h2>
                <form action="/get_text/" enctype="multipart/form-data" method="post">
                <input name="img" type="file">
                <input type="submit">
                </form>

                <h2>Templated PDF</h2>
                <form action="/pdf_template/" enctype="multipart/form-data" method="post">
                <input name="img" type="file">
                <input type="submit">
                </form>

                <h2>Add a template</h2>
                <form action="/add_template/" enctype="multipart/form-data" method="post">
                <input name="img" type="file">
                <input type="submit">
                </form>
                """

    return HTMLResponse(content=content)
