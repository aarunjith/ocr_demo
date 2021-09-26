__all__ = ['loaddill', 'register_to_template', 'read_content']

from torch_snippets import loaddill, cv2, np
import xml.etree.ElementTree as ET


def register_to_template(image, templ, return_homography=False):
    """register `image` to `templ`
    if `templ` is `np.ndarray` keypoints are computed from scratch, else are loaded as a dill file
    """
    detector = cv2.AKAZE_create()
    h, w = image.shape[:2]
    if isinstance(templ, np.ndarray):
        kps1, descs1 = detector.detectAndCompute(templ, None)
    else:
        kps1, descs1 = loaddill(templ)
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
