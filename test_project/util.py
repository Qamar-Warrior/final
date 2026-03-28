import string
import cv2
import numpy as np
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'U': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'B': '8'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '8': 'B'}

def license_complies_format(text):
    if len(text) != 8:
        return False

    def is_num(c):
        return c.isdigit() or c in dict_char_to_int.keys()
    def is_let(c):
        return c.isalpha() or c in dict_int_to_char.keys()

    # Region Code (01, 10, etc)
    if not (is_num(text[0]) and is_num(text[1])):
        return False

    # Type 1: 01 123 AAA
    if is_num(text[2]) and is_num(text[3]) and is_num(text[4]) and \
       is_let(text[5]) and is_let(text[6]) and is_let(text[7]):
        return True
    
    # Type 2: 01 A 123 AA
    if is_let(text[2]) and is_num(text[3]) and is_num(text[4]) and \
       is_num(text[5]) and is_let(text[6]) and is_let(text[7]):
        return True

    return False

def format_license(text):
    # Detect Type based on index 2
    is_type_2 = text[2].isalpha() or text[2] in dict_int_to_char.keys()
    
    if is_type_2: # 01 A 777 AA
        mapping = {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_int_to_char, 
                   3: dict_char_to_int, 4: dict_char_to_int, 5: dict_char_to_int, 
                   6: dict_int_to_char, 7: dict_int_to_char}
    else: # 01 777 AAA
        mapping = {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_char_to_int, 
                   3: dict_char_to_int, 4: dict_char_to_int, 5: dict_int_to_char, 
                   6: dict_int_to_char, 7: dict_int_to_char}

    res = ''
    for i in range(len(text)):
        if i in mapping and text[i] in mapping[i].keys():
            res += mapping[i][text[i]]
        else:
            res += text[i]
    return res

def preprocess_plate(img):
    """Return a list of preprocessed variants to improve OCR accuracy."""
    variants = []

    # Upscale for better OCR
    h, w = img.shape[:2]
    scale = max(1, 200 // h)
    if scale > 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants.append(img)          # original (resized)
    variants.append(gray)         # grayscale

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    variants.append(enhanced)

    # Otsu binarization
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # Inverted Otsu (dark text on light background)
    variants.append(cv2.bitwise_not(otsu))

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    variants.append(adaptive)

    return variants


def read_license_plate(license_plate_crop):
    best_text, best_score = None, 0.0

    for variant in preprocess_plate(license_plate_crop):
        detections = reader.readtext(variant)
        for detection in detections:
            bbox, text, score = detection
            text = text.upper().replace(' ', '').replace('-', '')
            if license_complies_format(text) and score > best_score:
                best_text = format_license(text)
                best_score = score

    if best_text is not None:
        return best_text, best_score
    return None, None

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    for vehicle in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle
        if xcar1 < center_x < xcar2 and ycar1 < center_y < ycar2:
            return vehicle
    return -1, -1, -1, -1, -1

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_score,license_number,license_number_score\n')
        for frame in results.keys():
            for car_id in results[frame].keys():
                data = results[frame][car_id]
                if 'license_plate' in data:
                    f.write(f"{frame},{car_id},{data['car']['bbox']},{data['license_plate']['bbox']},"
                            f"{data['license_plate']['bbox_score']},{data['license_plate']['text']},"
                            f"{data['license_plate']['text_score']}\n")