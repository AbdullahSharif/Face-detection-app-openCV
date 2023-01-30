import cv2

# get the trained data model

trained_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# read the video
web_cam = cv2.VideoCapture(0)       # if you give it 0 as an argument, it will take the video from your default webcam

while True:
    successful_frame_read, frame = web_cam.read()

    # convert the image to gray-scale.
    gray_scale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # now find the coordinates for the face
    face_coordinates = trained_data.detectMultiScale(gray_scale_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face detection app", frame)

    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

# now release the web_cam resources that we are using.
web_cam.release()