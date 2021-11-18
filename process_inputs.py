from tempfile import template
from pdf2image import convert_from_bytes
from loguru import logger
import numpy as np
from src.image_registration import *
from torch_snippets import read, crop_from_bb, Glob

def process_pdf(pdf_bytes, ocr):
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

    templates = Glob('./templates/*_resized.kp')

    logger.info('Processing PDF ....')
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
        crop_results = ocr.recognize(crop)[0]
        text = crop_results[1]
        conf = crop_results[2]
        result[name] = {"text": text, "confidence": conf, 'bb': box}
    logger.info('Transcription Complete.......')

    return output_img, result
