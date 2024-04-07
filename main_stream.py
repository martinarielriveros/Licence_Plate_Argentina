from ultralytics import YOLO
import cv2
from util_stream import draw_border, check_if_inside_light, license_complies_format_arg, format_license_arg, check_if_inside
from easyocr import Reader
import numpy as np
import pandas as pd

reader = Reader(['en'], gpu=False)

# Load models:  a "model" refers to the neural network architecture used for object detection tasks.
# Two models, one to detect cars and the other to detect licence plates
detector = YOLO('./models/yolov8n.pt') # Pre-Trained model using coco dataset (YOLO v8 nano)
license_plate_detector = YOLO('./models/license_plate_detector.pt') # Pre-Trained model using custom dataset

# load video

cap = cv2.VideoCapture('./videos/autopista.mp4')


# This are class_id from the dataset that we want to track (search inside the "training_config.yaml")
vehicles_categories = [2, 3, 5, 7] # Recognize cars, trucks, etc.

# 1) Read frames from video or stream
img = []
ret = True
texts_recognized = []

while ret:
    
    # Read frames from a video file or a camera stream. It returns two values
    # Boolean value (ret): This indicates whether the frame was successfully read or not
    # This is the actual frame read from the video file or camera stream.
    # If ret is True, frame will contain the image data of the frame; otherwise, frame may be empty or contain garbage data.
    
    ret, frame = cap.read()

    # Exit if 'q' is pressed, and wait a milisecond between runs for the screen to render (important)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(texts_recognized)
        break

    if ret:
         # 2) Detect vehicles - Run inference on an image

        # Calls will return a list of Results objects
        
        # REMEMBER: inferences will be executed FRAME by FRAME, so in a "live" streaming, frames will be lost
        # if the processing time lenghth of the image takes longer than the rate at wich the camera gets a new frame.
        # If a camera is 30 fps, the longest time for a frame to render (involving all your custom process) for 
        # frames not to be lost must be not grater than 1/30 = 0.0333 -> 33 ms
        
        detections = detector(frame)[0]
        
        # The detector "detector(frame)" creates an object of the form:
        
        # [ultralytics.engine.results.Results object with attributes:
        # boxes: ultralytics.engine.results.Boxes object
        # keypoints: None
        # masks: None
        # names: {0: 'license_plate'}
        # obb: None
        # orig_img: array([[[54, 49, 41],
        #         [54, 49, 41],
        #         [54, 49, 41],
        #         ...,
        #         [29, 26, 25],
        #         [28, 25, 24],
        #         [28, 25, 24]],
        #         --------------
        # orig_shape: (674, 1200)
        # path: 'image0.jpg'
        # probs: None
        # save_dir: 'runs\\detect\\predict'
        # speed: {'preprocess': 2.0012855529785156, 'inference': 129.99439239501953, 'postprocess': 0.9996891021728516}]

        # In the "boxes" key, there is another object with:

        # ultralytics.engine.results.Boxes object with attributes:

        # cls: tensor([0., 0.])
        # conf: tensor([0.4958, 0.4562])
        # data: tensor([[7.5379e+02, 4.9429e+02, 8.0906e+02, 5.1905e+02, 4.9577e-01, 0.0000e+00],
        #         [2.9401e+02, 5.5856e+02, 3.7273e+02, 5.9181e+02, 4.5616e-01, 0.0000e+00]])
        # id: None
        # is_track: False
        # orig_shape: (674, 1200)
        # shape: torch.Size([2, 6])
        # xywh: tensor([[781.4244, 506.6700,  55.2779,  24.7527],
        #         [333.3729, 575.1836,  78.7183,  33.2523]])
        # xywhn: tensor([[0.6512, 0.7517, 0.0461, 0.0367],
        #         [0.2778, 0.8534, 0.0656, 0.0493]])
        # xyxy: tensor([[753.7855, 494.2936, 809.0634, 519.0463],
        #         [294.0137, 558.5575, 372.7320, 591.8098]])
        # xyxyn: tensor([[0.6282, 0.7334, 0.6742, 0.7701],
        #         [0.2450, 0.8287, 0.3106, 0.8781]])

        # The "data" tensor is an interable and has the detections in the frame, ordered like:
        
        # (x_min, y_min, x_max, y_max, conf, cls) # Fist two are the top-left corrdinates, second two are the bottom-right

        custom_detections_ = [] # Store the bounding boxes coords, conf and cls of all vehicles detected
        licence_detections_ = [] # Store bounding boxes coords, conf and cls for detected licence plates
        
        for custom_detection in detections.boxes.data.tolist():

            x1, y1, x2, y2, conf, cls = custom_detection
            
            frame = draw_border(frame, int(x1), int(y1), int(x2), int(y2))
            if int(cls) in vehicles_categories:
                custom_detections_.append(custom_detection) # This is a list of all class_id matches for the detection
        

        # 3) Detect license plates
      
        license_plates = license_plate_detector(frame)[0]
     
        for license_plate in license_plates.boxes.data.tolist():

            licence_detections_.append(license_plate)
        

        # crop license plate from the original frame
        for licence_plate in  licence_detections_:


            x1,y1,x2,y2,_,_ = licence_plate
            frame = draw_border(frame, int(x1), int(y1), int(x2), int(y2))
            # int(y1):int(y2): This selects rows from index y1 to y2 (exclusive), representing the vertical range of the region of interest (ROI).
            # int(x1):int(x2): This selects columns from index x1 to x2 (exclusive), representing the horizontal range of the ROI.
            # : This selects all the color channels in the frame. In this case, since it's :, it implies selecting all color channels (e.g., RGB channels for a color image).

            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # process license plate

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    
            # All pixels lower than 64 are goint to be assigned to 255 and all pixels above 64 will be 0
            
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 60, 255, cv2.THRESH_BINARY_INV)

            # read license plate number
            
            # real_detection = reader.readtext(license_plate_crop_thresh, paragraph=True)
            # print(real_detection)
            detection_to_erase = reader.readtext(license_plate_crop_thresh, paragraph=False)
            print(detection_to_erase)
            # Detection has the form (paragraph=True):
            # [[[[5, 9], [54, 9], [54, 23], [5, 23]], 'LEYSV']]

            # Detection has the form (paragraph=False):
            # [[([[7, 19], [137, 19], [137, 61], [7, 61]], 'EFPB611', 0.18031569221904775)]
            try:

                # licence_plate_text = real_detection[0][-1].upper().replace(' ', '')
                licence_plate_text = detection_to_erase[0][1].upper().replace(' ', '')
                # Convert each inner list to a string and join them with a separator
                text_recognized_toshow = ' '.join([' '.join(map(str, item)) for item in texts_recognized])
                
                # text = str(licence_plate_text) + text_recognized_toshow 
                #--------------
                # licence_plate_text has the "AF 986 MS" format. Not "corrected".


                # Define a kernel for morphological operation
                kernel = np.ones((2,2), np.uint8)  # Adjust the size of the kernel as needed

                # Apply erosion to make lines thinner
                license_plate_crop_thresh = cv2.erode(license_plate_crop_thresh, kernel, iterations=1)
                
                height1, width1 = license_plate_crop_thresh.shape
                height = 100
                width = 100
                # Calculate the position for the text
                text_position = (int(width1 * 0.5), int(height1 * 0.5))
                resized_image = cv2.resize(license_plate_crop_thresh, (height*2, width*1))
                crop_licence = cv2.putText(resized_image, text_recognized_toshow, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.imshow('licence', crop_licence)
                
                # -----------------------------
                
                if license_complies_format_arg(licence_plate_text):
                    print('licence before formatting :', licence_plate_text)
                    licence_formatted = format_license_arg(licence_plate_text)
                    print('licence after formatting :', licence_formatted)
                    # Here we send ONE licence_plate (the one that has "good" shape and "format") and ALL custom_detections_ (E.g: cars)
        
                    licence_vehicle_match = check_if_inside_light(licence_plate, custom_detections_)
                    
                    licence_info= licence_vehicle_match[0]
                    vehicle_position = licence_vehicle_match[1]
                    
                    
                    frame = draw_border(frame, int(licence_info[0]),int(licence_info[1]), int(licence_info[2]), int(licence_info[3]))
                    
                    if len(licence_vehicle_match) != 0:
                        
                        # licence_ and vehicle has the coords of the licence and vehicle where they match
                        
                        isIn = False
                        already_appended = False

                        if len(texts_recognized)==0:
                            texts_recognized.append([licence_formatted, licence_info[4]])

                        for licence in texts_recognized:
                            print(licence[0])
                            if licence[0] == licence_formatted:
                                isIn = True
                        
                        for licence in texts_recognized:
                            if isIn and not already_appended:
                                
                                if licence[0] == licence_formatted and licence[1] < licence_info[4]:
                                    texts_recognized[-1] = [licence_formatted, licence_info[4]]
                                    already_appended = True
                            
                            elif not isIn and not already_appended:
                                texts_recognized.append([licence_formatted, licence_info[4]])
                                isIn = True
                                already_appended=True


                        # Define the position of the text
                        text_position = (int(licence_info[0]) + 8, int(licence_info[1]) - 15)

                        # Get the text size
                        text_size, _ = cv2.getTextSize(licence_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 3)

                        # Calculate the bounding box for the background rectangle
                        background_box = ((text_position[0], text_position[1] - text_size[1]), 
                                        (text_position[0] + text_size[0], text_position[1]))

                        # Draw the white background rectangle
                        cv2.rectangle(frame, background_box[0], background_box[1], (255, 255, 255), cv2.FILLED)

                        # Draw the text on the image
                        cv2.putText(frame, licence_plate_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    else:
                        print('licence plate outside vehicle')
                        frame = draw_border(frame, int(licence_info[0]),int(licence_info[1]), int(licence_info[2]), int(licence_info[3]))            
                else:
                    print('licence plate does not comply format', licence_plate_text)

            except Exception as e:
                print('Error :', e)

    cv2.imshow('Final', frame)
# Convert the list to a DataFrame
results = pd.DataFrame(texts_recognized, columns=['Text', 'Value'])

# Export the DataFrame to a CSV file
results.to_csv('results.csv', index=False)
cap.release()
cv2.destroyAllWindows()
        
  





  


