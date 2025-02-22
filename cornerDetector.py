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

# --------------------------------------------------------------------- #

# Create a custom kernel

# 3x3 array for corner detection
topright = np.array([[  0,  1,  1],
                    [  0,  1,  1],
                    [  0,  0,  0]])

## TODO: Create and apply a vertical edge detection operator
topleft = np.array([[ 1, 1, 0],
                    [ 1, 1, 0],
                    [ 0, 0, 0]])

bottomleft = np.array([[ 0, 0, 0],
                    [ 1,  1, 0],
                    [  1,  1, 0]])

bottomright = np.array([[ 0, 0, 0],
                    [ 0,  1, 1],
                    [ 0,  1,  1]])

fig = plt.figure(figsize=(24,24))
# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image = cv2.filter2D(gray, -1, topright)
fig.add_subplot(2,2,1)
plt.imshow(filtered_image, cmap='gray')
plt.title('topright')

filtered_image2 = cv2.filter2D(gray, -1, topleft )
fig.add_subplot(2,2,2)
plt.imshow(filtered_image2, cmap='gray')
plt.title('topleft')

filtered_image3 = cv2.filter2D(gray, -1, bottomleft)
fig.add_subplot(1,2,1)
plt.imshow(filtered_image3, cmap='gray')
plt.title('bottomleft')

filtered_image4 = cv2.filter2D(gray, -1, bottomright)
fig.add_subplot(1,2,2)
plt.imshow(filtered_image4, cmap='gray')
plt.title('bottomright')

plt.show()
plt.savefig("CornerDetector.png")