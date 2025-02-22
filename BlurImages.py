import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

# Read in the image
image = mpimg.imread('building2.jpg')

plt.imshow(image)

# Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')

# 2x2 kernel for averaging blurring
S2x2 = np.array([[ 1, 1],
                 [ 1, 1]])

S3x3 = np.array([[ 1, 1, 1],
                 [ 1, 1, 1],
                 [ 1, 1, 1]])

S5x5 = np.array([[ 1, 1, 1, 1, 1],
                 [ 1, 1, 1, 1, 1],
                 [ 1, 1, 1, 1, 1],
                 [ 1, 1, 1, 1, 1],
                 [ 1, 1, 1, 1, 1]])

fig = plt.figure(figsize=(48, 12))
fig.add_subplot(4,1,1)
plt.imshow(gray, cmap='gray')
plt.title('original')

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
blurred_image = cv2.filter2D(gray, -1, S2x2/4.0)
fig.add_subplot(4,1,2)
plt.imshow(blurred_image, cmap='gray')
plt.title('2x2')

# TODO: blur image using a 3x3 average
blurred_image = cv2.filter2D(gray, -1, S3x3/9.0)
fig.add_subplot(4,1,3)
plt.imshow(blurred_image, cmap='gray')
plt.title('3x3')

# TODO: blur image using a 5x5 average
blurred_image = cv2.filter2D(gray, -1, S5x5/25.0)
fig.add_subplot(4,1,4)
plt.imshow(blurred_image, cmap='gray')
plt.title('5x5')

plt.show()
plt.savefig("BlurFilters.png")