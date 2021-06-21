import cv2, glob, imutils
from imutils.convenience import resize
import numpy as np

from pre_processing import get_plate_bbox_and_roi, get_char_bboxes_and_rois
from matching import compare_characters, draw_results
    

if __name__ == '__main__':
    image = cv2.imread('placa.jpg', cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread('placa.jpg')
    img = image.copy()
    plate_bbox, plate_roi = get_plate_bbox_and_roi(image)
    char_bboxes, char_rois = get_char_bboxes_and_rois(plate_roi, plate_bbox)
    result = compare_characters(char_rois, char_bboxes)
    draw_results(result, color_image)