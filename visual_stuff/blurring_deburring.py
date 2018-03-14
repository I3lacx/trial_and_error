import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5,5),np.float32)/25
    kernel2 = np.ones((3,3),np.float32)/9
    blurred = cv2.filter2D(gray,-1,kernel)
    blurred2 = cv2.filter2D(gray,-1,kernel2)

    denoised = cv2.fastNlMeansDenoising(gray)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    cv2.imshow('blurred', blurred)
    cv2.imshow('blurred2', blurred2)
    cv2.imshow('denoised', denoised)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
