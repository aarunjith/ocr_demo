from uuid import uuid1
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from easyocr import Reader
from src.image_utils import convert_bbs
import numpy as np
from loguru import logger
from starlette.responses import FileResponse, JSONResponse
from process_inputs import process_pdf
import cv2
from uuid import uuid1

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


@app.get('/image/{impath}')
async def show_image(impath: str):
    return FileResponse(f'data/{impath}')


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
                """

    return HTMLResponse(content=content)
