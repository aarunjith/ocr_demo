from loguru import logger
import numpy as np
from pdf2image.pdf2image import convert_from_path, convert_from_bytes
from src.image_registration import detect_template, read_content, register_to_template
from torch_snippets import read, crop_from_bb, Glob
import boto3
from PIL import Image
from io import BytesIO

def process_pdf(pdf_bytes, ocr, textract=False):
    '''
    Transcribe input pdf (bytes) after the following steps:
        1. Registers the first page to the template image
        2. Remove labels from fields

    Inputs:
        pdf_bytes : bytes -> Bytes of a pdf file recieved through an API
        ocr : EasyOCR Reader Instance -> EasyOCR reader instance to be used for
              transcription

    Outputs:
        Python Dict -> Dictionary containing Key Value pairs extracted from the pdf
    '''

    templates = Glob('./templates/*.xml')
    templates = [str(template) for template in templates]
    templates = [temp.split('.xml')[0]+'_resized.kp' for temp in templates]
    if isinstance(pdf_bytes, str):
        logger.info('Processing PDF path....')
        ims = convert_from_path(pdf_bytes)
        first_page = np.array(ims[0])
    elif isinstance(pdf_bytes, np.ndarray):
        first_page = pdf_bytes
    else:
        logger.info('Processing PDF bytes....')
        ims = convert_from_bytes(pdf_bytes)
        first_page = np.array(ims[0])

    logger.info('Detecting template')
    template_kps, template_xml = detect_template(first_page, templates)
    if template_kps == -1:
        logger.info('No valid template found.....')
        return first_page, {'error': "Not enough matches"}
    impath, names, boxes = read_content(template_xml)
    label_boxes = [box for box, name in zip(boxes, names) if name == 'label']
    field_boxes = [(box, name)
                   for box, name in zip(boxes, names) if name != 'label']

    logger.info('Registering to template.......')
    registered_img = register_to_template(first_page, template_kps)
    logger.info('Removing labels.......')

    output_img = registered_img
    for label_box in label_boxes:
        x, y, X, Y = label_box
        registered_img[y:Y, x:X] = 255
    result = {}
    logger.info('Transcribing.......')
    for box, name in field_boxes:
        crop = crop_from_bb(registered_img, tuple(box))
        if textract:
            client = boto3.client('textract', region_name='eu-west-2')
            text, conf = get_text_textract(crop, client)
            result[name] = {"text": text, "confidence": conf, 'bb': box}
        else:
            crop_results = ocr.recognize(crop)[0]
            text = crop_results[1]
            conf = crop_results[2]
            result[name] = {"text": text, "confidence": conf, 'bb': box}
    logger.info('Transcription Complete.......')

    return output_img, result

def get_text_textract(image, client):
    '''
    image : Numpy array image
    '''
    image = Image.fromarray(image)
    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()
    client = boto3.client('textract', region_name='eu-west-2')
    response = client.detect_document_text(Document={'Bytes': image_bytes})
    text = [{'text':doc['Text'], 'conf': doc['Confidence']} for doc in response['Blocks'] if doc['BlockType'] == 'LINE']
    text = {'text': '\n'.join([doc['text'] for doc in text]), 'conf': min([doc['conf'] for doc in text])}
    return text['text'], text['conf']
