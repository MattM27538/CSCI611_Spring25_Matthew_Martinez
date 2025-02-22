import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

# Read in the image
image = mpimg.imread('image_filtering/building2.jpg')


halfScale = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)
plt.imshow(halfScale)
plt.savefig("testHalfScale.png")

# plt.clf()
quarterScale = cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25)
plt.imshow(quarterScale)
plt.savefig("testQuarterScale.png")
