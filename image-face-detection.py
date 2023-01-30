import cv2


"""
    General algorith we will be using.
    -> first get the trained data set.
    -> read the colored images.
    -> conver the images to black and white(gray-scale)
    -> apply the trained algorith on it.
"""

# load the pre-trained data.
trained_face_data = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")

# now we need to read the image.
image_read = cv2.imread("RDJ.png")


# convert the image to grayscale.
gray_sclae_img = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)

# now detect teh face.
face_coordinates = trained_face_data.detectMultiScale(gray_sclae_img)
print(face_coordinates)

# now draw the rectangle on the image.
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(image_read, (x, y), (x+ w, y+h), (0, 255, 0), 2)

# show the image.
cv2.imshow("Face detection app", image_read)

# the following function will hold the window displaying the image until any key is pressed.
cv2.waitKey()


print("Code completed!")
