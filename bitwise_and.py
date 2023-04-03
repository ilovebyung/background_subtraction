import cv2
import matplotlib.pyplot as plt
import numpy as np

fg = './objects/fg/2.jpeg'
bg = './objects/bg/1.jpeg'

bg = cv2.imread(bg, 0)
# plt.imshow(bg, cmap='gray')
fg = cv2.imread(fg, 0)
# plt.imshow(fg, cmap='gray')

mask = cv2.absdiff(fg, bg)

threshold = 10
imask = mask > threshold

############ bg #############
canvas = np.zeros_like(bg, np.uint8)
canvas[imask] = bg[imask]

plt.imshow(canvas, cmap='gray')

############ fg #############
canvas = np.zeros_like(fg, np.uint8)
canvas[imask] = fg[imask]

plt.imshow(canvas, cmap='gray')

############ absdiff #############
diff = cv2.absdiff(fg, bg)
plt.imshow(diff, cmap='gray')

############ minus #############


merge = cv2.merge([fg, bg, bg])
plt.imshow(merge)
