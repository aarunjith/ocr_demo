__all__ = ['loaddill', 'register_to_template',
           'read_content', 'detect_template', 'dump_template']

from torch_snippets import loaddill, cv2, np, logger, dumpdill
import xml.etree.ElementTree as ET
from PIL import Image
from typing import Union
from xml.dom.minidom import parseString

filepath = __file__
TEMPLATES = '/'.join(filepath.split('/')[:-2])+'/templates'


def dump_keypoints(kps, desc, filename):
    '''
    Preprocesses kepypoints and dumps them onto the disk
    Inputs:
        kps, desc -> Keypoints and Description computed using AKAZE
    '''
    pickle_kps = []
    for point in kps:
        kp = (point.pt, point.size, point.angle, point.response, point.octave,
              point.class_id)
        pickle_kps.append(kp)
    dumpdill((pickle_kps, desc), filename)


def dump_template(image, filename):
    '''
    Computes the keypoints for the original and resized template image and dumps them onto disk along with the image file
    Inputs:
        image -> Input image(usually that of a blank template) as a Numpy Array
        filename -> Filename to be used to identify the keypoints
    '''
    detector = cv2.AKAZE_create()
    kps, desc = detector.detectAndCompute(image, None)
    kps_resized, desc_resized = detector.detectAndCompute(
        image[::4, ::4], None)
    Image.fromarray(image).save(f'{TEMPLATES}/{filename}.jpg')
    dump_keypoints(kps, desc, f'{TEMPLATES}/{filename}.kp')
    dump_keypoints(kps_resized, desc_resized,
                   f'{TEMPLATES}/{filename}_resized.kp')
    logger.info('DONE!')


def load_keypoints(dill):
    '''
    Loads cv2.KeyPoints from a pickle file as pickling takes an additional preprocessing step
    Inputs:
        dill: str -> Path location to the file dumped using dill 
    '''
    kps, descs = loaddill(dill)
    key_points = []
    for point in kps:
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2],
                                    response=point[3], octave=point[4], class_id=point[5])
        key_points.append(temp_feature)
    return key_points, descs


def get_matches(image, templ):
    """
    Compute the number of good matches between the image and template
        Inputs:
            image : np.ndarray -> Image that needs to be identified
            templ : str/np.ndarray -> Template image or saved keypoints for the template
        Outputs:
            good : Integer -> Number of good matches
    """
    detector = cv2.AKAZE_create()
    if isinstance(templ, np.ndarray):
        kps1, descs1 = detector.detectAndCompute(templ, None)
    else:
        kps1, descs1 = load_keypoints(templ)
    kps2, descs2 = detector.detectAndCompute(image, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1, descs2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    return len(good)


def detect_template(image, templates):
    '''
    Detect the template using highest number of good matches using get_matches
        Inputs:
            image : np.ndarray -> Image that needs to be identified
            templates : List (str/ np.ndarray) -> Template images or saved 
                                                  keypoints for the templates
        Outputs:
            template : Identified template
    '''
    matches = []
    for templ in templates:
        good = get_matches(image[::4, ::4], templ)
        matches.append(good)
    max_matches = np.argmax(matches)
    if matches[max_matches] < 20:
        return -1, -1
    template_prefix = templates[max_matches].split("resized")[0].strip("_")
    logger.info(template_prefix)
    return template_prefix + '.kp', template_prefix + '.xml'


def register_to_template(image, templ, return_homography=False):
    """register `image` to `templ`
    if `templ` is `np.ndarray` keypoints are computed from scratch, else are loaded as a dill file
    """
    detector = cv2.AKAZE_create()
    h, w = image.shape[:2]
    if isinstance(templ, np.ndarray):
        kps1, descs1 = detector.detectAndCompute(templ, None)
    else:
        kps1, descs1 = load_keypoints(templ)
    kps2, descs2 = detector.detectAndCompute(image, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1, descs2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])
    ref_matched_kpts = np.float32([kps1[m[0].queryIdx].pt for m in matches])
    sensed_matched_kpts = np.float32([kps2[m[0].trainIdx].pt for m in matches])
    _H_, status = cv2.findHomography(
        sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC, 5.0)
    warped_image = cv2.warpPerspective(
        image, _H_, (w, h), borderValue=[int(np.median(image))]*3)
    # warped_image = warped_image[0:H,0:W]
    if return_homography:
        return warped_image, _H_
    else:
        return warped_image


def read_content(xml_file: str):
    '''
    Reads ground truth bounding boxes from PascalVOC formatted xmls
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    box_names = []

    filepath = root.find('filename').text
    for boxes in root.iter('object'):

        box_name = boxes.find('name').text
        box_names.append(box_name)

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filepath, box_names, list_with_all_boxes


def object_to_xml(data: Union[dict, bool], root='annotation'):
    '''
    Convert python object to XML string
    '''
    if root.startswith('object'):
        root = 'object'
    xml = f'<{root}>'
    if isinstance(data, dict):
        for key, value in data.items():
            xml += object_to_xml(value, key)

    else:
        xml += str(data)

    xml += f'</{root}>'
    return xml


def convert_to_pascal(annotation, filename):
    '''
    Convert Label Studio Annotation to PASCAL VOC format.
    Inputs:
        annotation -> List of dicts containing annotation information
    '''
    width = annotation[0]['original_width']
    height = annotation[0]['original_height']
    annotation_dict = {'filename': f'{filename}.jpg'}
    for ix, ob in enumerate(annotation):
        annotation_dict[f'object_{ix}'] = {'name': ob['value']['rectanglelabels'][0], 'bndbox': {
            'xmin': int(ob['value']['x']/100*width),
            'ymin': int(ob['value']['y']/100*height),
            'xmax': int(ob['value']['x']/100*width) + int(ob['value']['width']/100*width),
            'ymax': int(ob['value']['y']/100*height) + int(ob['value']['height']/100*height)
        }}
    xml = object_to_xml(annotation_dict)
    dom = parseString(xml)
    logger.info(f'Writing XML data to {TEMPLATES}/{filename}.xml')
    with open(f'{TEMPLATES}/{filename}.xml', 'w') as doc:
        doc.write(dom.toprettyxml())
    logger.info(f'DONE!!')
