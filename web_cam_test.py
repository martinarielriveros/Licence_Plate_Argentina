import cv2

cap = cv2.VideoCapture(1)
ret=True

while ret:

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()
    cv2.imshow("web cam", frame)