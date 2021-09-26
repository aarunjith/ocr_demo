from scipy.ndimage import interpolation as inter
from torch_snippets import np
import cv2


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


def skew_angle(im, delta=0.5, limit=5):
    """
    Computes the skew angle given a rotated image
    Args:
        im : Numpy Array -> Image to compute the angle on, needs to be binary
        delta : Step delta between angles. Smaller values will take more compute
        limit : Limit of the angle of rotation i.e rotation angle will be [-limit, limit+delta]

    Returns:
        float -> Best angle computed for derotation
    """

    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(im, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    return best_angle


def derotate_img(im, shrinkfactor=2, angle=None):
    """
    Derotate an image by autocomputing the rotation angle
    Args:
        im (np.ndarray): Image to unrotate
        shrinkfactor (int): Shrink image's dimensions before computing rotation
        Useful as large images take longer time to process
        angle (float, None): If angle is give, use that to detorate instead of computing
    Returns:
        np.ndarray: Rotated image
    """
    sf = shrinkfactor
    angle = skew_angle(im[::sf, ::sf] > 127) if angle is None else angle
    return inter.rotate(im, angle, reshape=True, order=0, cval=255)


def rotate_image(image, angle):
    '''
    Rotate the input image by the input angle
    Args:
        image: np.ndarray -> Input image array to be rotated
        angle: float -> Angle of rotation

    Returns:
        np.ndarray -> Rotated image array
    '''
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def convert_bbs(bb):
    '''
    Convert bb form (xi,yi) format to (x1,y1,x2,y2) for easy decoding
    Args:
        bb : Bounding box in the format [[x1,y1], [x2, y2], [x3,y3], [x4,y4]]
    Returns:
        List(tuples) -> Converted bb in the new format
    '''
    assert isinstance(bb, list)
    x1, y1 = bb[0]
    x2, y2 = bb[2]
    bb_new = list(map(int, [x1, y1, x2, y2]))
    return bb_new
