import cv2

def draw_border(img, x1,y1,x2,y2, color=(0, 255, 0), thickness=1):

    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    return img