import cv2
import matplotlib.pyplot as plt
import glob

files = glob.glob('./objects/*jpeg')

# Load the image and the background
image = cv2.imread(files[0], 0)
background = cv2.imread(files[1], 0)

plt.imshow(image, cmap='gray')
plt.imshow(background, cmap='gray')

# Subtract the background from the image
subtracted = cv2.absdiff(image, background)
plt.imshow(subtracted, cmap='gray')

# Apply thresholding to the subtracted image
thresh = cv2.threshold(subtracted, 25, 255, cv2.THRESH_BINARY)[1]

# Display the result
plt.imshow(thresh, cmap='gray')

cv2.imshow('Subtracted image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
1.Background Initialization: an initial model of the background is computed
'''

'''
2.Background Update: model is updated in order to adapt to possible changes in the scene
'''
