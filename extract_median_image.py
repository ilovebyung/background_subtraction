import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

'''
get background from images 
'''
files = glob.glob('./objects/bg/*jpeg')

# creating an array of frames from frames chosen above
frames = []
for file in files:
    frame = cv2.imread(file, 0)
    frames.append(frame)

# calculate the median
bg = np.median(frames, axis=0).astype(dtype=np.uint8)
plt.imshow(bg, cmap='gray')
# cv2.imwrite("background.jpg", bg)


'''
get foreground from images 
'''
files = glob.glob('./objects/fg/*jpeg')

# creating an array of frames from frames chosen above
frames = []
for file in files:
    frame = cv2.imread(file, 0)
    frames.append(frame)

# calculate the mean
fg = np.mean(frames, axis=0).astype(dtype=np.uint8)
plt.imshow(fg, cmap='gray')
# cv2.imwrite("background.jpg", bg)

cv2.imshow('fg', fg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Subtract the background from the image
subtracted = cv2.absdiff(fg, bg)
plt.imshow(subtracted, cmap='gray')
