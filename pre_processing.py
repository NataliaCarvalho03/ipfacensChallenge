import cv2
import numpy as np


def pre_process_image(image):
    blurred = cv2.medianBlur(image, 11)
    ret,thresh1 = cv2.threshold(blurred,175,255,cv2.THRESH_BINARY)
    kernel = np.ones((7, 7),np.uint8)
    eroded = cv2.erode(thresh1,kernel,iterations = 1)
    return eroded


def pre_process_characters(image):
    blurred = cv2.medianBlur(image, 11)
    ret,thresh = cv2.threshold(blurred,175,255,cv2.THRESH_BINARY_INV)
    return thresh


def pre_process_template(template_image):
    template_image = cv2.resize(template_image, (42,86))
    inverted = cv2.bitwise_not(template_image)
    processed_template = cv2.dilate(inverted, np.ones((2,2)))
    return processed_template


def get_plate_bbox_and_roi(image):
    box = None
    image_roi = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    image_total_area = image.shape[0] * image.shape[1]
    processed = pre_process_image(image)
    contours,_=cv2.findContours(processed, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= int(0.05 * image_total_area) and area < int(0.10 * image_total_area):
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #cv2.drawContours(image,[box],0,(0,0,255),2)
    plate_crop = image[np.amin(box[:,1]):np.amax(box[:,1]), np.amin(box[:,0]):np.amax(box[:,0])]
    image_roi[np.amin(box[:,1]):np.amax(box[:,1]), np.amin(box[:,0]):np.amax(box[:,0])] = plate_crop
    cv2.imshow('Plate ROI', image_roi)
    cv2.waitKey(0)
    return [box, image_roi]


def get_char_bboxes_and_rois(image, bbox):
    boxes = []
    plate_crop = image[np.amin(bbox[:,1]):np.amax(bbox[:,1]), np.amin(bbox[:,0]):np.amax(bbox[:,0])]
    image_roi = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    processed_roi = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    processed = pre_process_characters(plate_crop)
    processed_roi[np.amin(bbox[:,1]):np.amax(bbox[:,1]), np.amin(bbox[:,0]):np.amax(bbox[:,0])] = processed
    image_total_area = image.shape[0] * image.shape[1]
    contours,_=cv2.findContours(processed_roi, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= int(0.0033 * image_total_area) and area <= int(0.01 * image_total_area):
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append(box)
            crop = image[np.amin(box[:,1]):np.amax(box[:,1]), np.amin(box[:,0]):np.amax(box[:,0])]
            ret,crop = cv2.threshold(crop,175,255,cv2.THRESH_BINARY_INV)
            image_roi[np.amin(box[:,1]):np.amax(box[:,1]), np.amin(box[:,0]):np.amax(box[:,0])] = crop
            cv2.drawContours(image,[box],0,(0,0,255),2)
    cv2.imshow('Chars ROI', image)
    cv2.waitKey(0)
    return [boxes, image_roi]