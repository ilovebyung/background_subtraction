import imutils
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# combine background & foreground images into out.mp4
# ffmpeg -pattern_type glob -i '*.jpeg' out.mp4
os.chdir('d:/source/lighting/objects/bg')
cap = cv2.VideoCapture("out.mp4")
# pBackSub = cv2.createBackgroundSubtractorMOG2()
pBackSub = cv2.createBackgroundSubtractorKNN()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgMask = pBackSub.apply(frame)

    # cv2.imshow("Frame", frame)
    cv2.imshow("FG Mask", fgMask)
    if cv2.waitKey(10) == 27:
        break

# cv2.imwrite('mask.jpg', fgMask)
cap.release()
cv2.destroyAllWindows()

plt.imshow(fgMask, cmap='gray')

# 1.Erosion
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(fgMask, kernel, iterations=1)
plt.imshow(erosion, cmap='gray')

# 2.Closing
closing = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing, cmap='gray')

# 3.Dilation
dilation = cv2.dilate(fgMask, kernel, iterations=1)
plt.imshow(dilation, cmap='gray')

# 4.Otsu's thresholding
ret, threshold = cv2.threshold(
    fgMask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(threshold, cmap='gray')

'''
combined threshold
'''
# 1.Erosion
kernel = np.ones((20, 20), np.uint8)
erosion = cv2.erode(fgMask, kernel, iterations=1)
# 2.Closing
closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing, cmap='gray')
# 3.Dilation
dilation = cv2.dilate(closing, kernel, iterations=1)
plt.imshow(dilation, cmap='gray')
# 4.Otsu's thresholding
ret, threshold = cv2.threshold(
    dilation, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(threshold, cmap='gray')

'''
background, foreground
'''
# Load image, create mask, and draw white circle on mask
mask = threshold
image = cv2.imread('d:/source/lighting/objects/bg/99.jpeg', 0)

# Mask input image with binary mask
result = cv2.bitwise_and(image, mask)
# Color background white
result[mask < 100] = 255  # Optional
plt.imshow(result, cmap='gray')

'''
contour approximation
'''

# find the largest contour in the threshold image
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
# draw the shape of the contour on the output image, compute the
# bounding box, and display the number of points in the contour
output = mask.copy()
cv2.drawContours(output, [c], -1, (0, 255, 0), 3)
(x, y, w, h) = cv2.boundingRect(c)
text = "original, num_pts={}".format(len(c))
cv2.putText(output, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2)
plt.imshow(mask, cmap='gray')


# show the original contour image
print("[INFO] {}".format(text))
cv2.imshow("Original Contour", output)
cv2.waitKey(0)
