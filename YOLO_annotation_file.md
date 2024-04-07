The YOLO dataset annotations are files (one for each image) referring to the bounding box of the licence plate.
The configuration must be decided on download of the datset from Roboflow site.

Once downloaded, each file reffering an image is a list like:

    0 0.5 0.4 0.3 0.6

    1 0.3 0.7 0.2 0.4

<**object-class**> <**x**> <**y**> <**width**> <**height**>


Each line of the annotation file:

**object-class**: An integer representing the class label of the detected object. For example, if you have 3 classes (person, car, and dog), these could be represented as 0, 1, and 2 respectively.

**x**: The x-coordinate of the center of the bounding box, normalized by the width of the image. For example, if the center of the bounding box is at 300 pixels in a 600-pixel wide image, the value would be 0.5 (300/600).

**y**: The y-coordinate of the center of the bounding box, normalized by the height of the image. Similarly, if the center of the bounding box is at 200 pixels in a 400-pixel high image, the value would be 0.5 (200/400).

**width**: The width of the bounding box, normalized by the width of the image. For example, if the bounding box width is 100 pixels in a 600-pixel wide image, the value would be approximately 0.1667 (100/600).

**height**: The height of the bounding box, normalized by the height of the image. Similarly, if the bounding box height is 80 pixels in a 400-pixel high image, the value would be 0.2 (80/400).