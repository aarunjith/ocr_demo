__all__ = ['loaddill', 'register_to_template',
           'read_content', 'detect_template']

from torch_snippets import loaddill, cv2, np, logger
import xml.etree.ElementTree as ET


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
