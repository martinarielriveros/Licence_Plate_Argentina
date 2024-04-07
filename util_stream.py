import string
import easyocr
import cv2

# Initialize the OCR reader to specific font (Argentina Mercosur)
# EasyOCR will then check if you have necessary model files and download them automatically.
# It will then load model into memory which can take a few seconds depending on your hardware.
 #After it is done, you can read as many images as you want without running this line again.

reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5', 
                    'B': '8'} # Added 'B': '8'

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '8': 'B'} # Added '8': 'B'

def license_complies_format_arg(text):
    """
    Check if the license plate text complies with the required format for Argentina Mercosur
    - 2 letters - 3 numbers - 2 letters and it's possible mistakes in recognition
    - 7 digit text

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False
    
    elif \
        (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
        (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
        (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
        (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
        (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
        (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):

        return True
    else:
        return False

def format_license_arg(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char,
               1: dict_int_to_char,
               2: dict_char_to_int,
               3: dict_char_to_int,
               4: dict_char_to_int,
               5: dict_int_to_char,
               6: dict_int_to_char}
    
    # Iterates over the 7 characters recognized, and compared to type

    for position in [0, 1, 2, 3, 4, 5, 6]:
        if text[position] in mapping[position].keys():
            license_plate_ += mapping[position][text[position]]
        else:
            license_plate_ += text[position]

    return license_plate_

def check_if_inside(licence, custom_detections):
    
    licence_=[]
    vehicle_=[]
    x1, y1, x2, y2, _, _ = licence
    
    for vehicle in custom_detections:
            
            vex1, vey1, vex2, vey2, _, _ = vehicle


            # This checks if the "y" license plate coords are below half of height and "x" points
            # are both sides of the half width.

            if  x1>vex1 and x1<vex1+(vex2-vex1)/2 and x2>vex1+(vex2-vex1)/2 and x2<vex2 \
                and y1>vey1+(vey2-vey1)/2 and y2>vey1+(vey2-vey1)/2 and y2<vey2 and y1<vey2:
                               
                licence_.append(licence)
                vehicle_.append(vehicle)
                
    
    return licence_[0], vehicle_[0]

def check_if_inside_light(licence, custom_detections):
    
    licence_=[]
    vehicle_=[]
    x1, y1, x2, y2, _, _ = licence
    
    for vehicle in custom_detections:
            
            vex1, vey1, vex2, vey2, _, _ = vehicle

            if  x1>vex1 and x2<vex2 and y1>vey1 and y2<vey2 :
                               
                licence_.append(licence)
                vehicle_.append(vehicle)
                  
    return licence_[0], vehicle_[0]


def draw_border(img, x1,y1,x2,y2, color=(0, 255, 0), thickness=1):

    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    return img