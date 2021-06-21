import cv2, imutils, glob
import numpy as np

from pre_processing import pre_process_template


def template_match(template, image, template_digit: str):
    digit_info = {'digit': template_digit, 'max_val': None, 'bbox': None}
    (tH, tW) = template.shape[:2]
    found = None
    for scale in np.linspace(0.2, 2.0, 20)[::-1]:
        resized = imutils.resize(image, width = int(image.shape[1] * scale))
        r = image.shape[1] / float(resized.shape[1])
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    digit_info['max_val'] = found[0]
    return digit_info


def get_most_confident_char(coefficients: list):
    max_values = [np.amax(data['max_val']) for data in coefficients]
    digits = [data['digit'] for data in coefficients]
    max_values = [np.amax(data['max_val']) for data in coefficients]
    max_coeff = max(max_values)
    if max_coeff > 0:
        most_conf_digit = digits[max_values.index(max_coeff)]
        return [most_conf_digit, max_coeff]
    return ["", max_coeff]


def compare_characters(characters_image, char_boxes):
    base_char_path = 'caracteres/'
    base_chars = glob.glob(base_char_path + '*.png')
    chars_metrics = []
    best_candidates = []
    for char_box in char_boxes:        
        img = characters_image[np.amin(char_box[:,1]):np.amax(char_box[:,1]), np.amin(char_box[:,0]):np.amax(char_box[:,0])]
        for template in base_chars:            
            temp = cv2.imread(template, 0)
            final_temp = pre_process_template(temp)
            char_info = template_match(final_temp, img, template.split('.')[0][-1])
            chars_metrics.append(char_info)
        char_metrics = get_most_confident_char(chars_metrics)
        char_metrics.append(char_box)
        best_candidates.append(char_metrics)
    return best_candidates


def draw_results(results_list, original_image):
    for result in results_list:
        cv2.drawContours(original_image, [np.int0(result[2])], 0, (0, 0, 255), 2)
        cv2.putText(original_image, f'{result[0]}', (result[2][0][0], result[2][0][1]-3),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow('Final Result', original_image)
    cv2.waitKey(0)