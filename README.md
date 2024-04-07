This is a Model for License Plate recognition, based on the approach taken from:

    https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8

![sample](https://github.com/martinarielriveros/Licence_Plate_Argentina/blob/master/videos/sample.png)


**Key differences:**

- Licence plate setup is modified to match Argentina - Mercosur digits sequence.


- No tracking for vehicles.
- Bounding boxes for license plates inside vehicles are restricted (not necesary though).
- Conversion digit added.

![template](https://github.com/martinarielriveros/Licence_Plate_Argentina/blob/master/videos/template.jpeg)

This was my first hobby-project with openCV.

**ISSUES:**

- Sample video is taken from behind, better if front side shoot.
- Initially Thought for live video, and as i got no GPU on my laptop, it took me too long to bug fix.


**TODO:**

- Add tracking to get best inference for a licence plate.
- Remove duplicated random detections from YOLO detector.
- Enhance licence crop visualization and results store.
