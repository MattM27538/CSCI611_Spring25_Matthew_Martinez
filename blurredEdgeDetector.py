import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

# Read in the image
image = mpimg.imread('park.jpg')


# Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


S3x3 = np.array([[ 1, 1, 1],
                 [ 1, 1, 1],
                 [ 1, 1, 1]])

vertica = np.array([[ -1, 0, 1],
                    [ -2, 0, 2],
                    [ -1, 0, 1]])


fig = plt.figure(figsize=(24,24))

blurred_image = cv2.filter2D(image, -1, S3x3/9.0)

filtered_image = cv2.filter2D(blurred_image, -1, vertica)
fig.add_subplot(1,2,1)
plt.imshow(filtered_image)
plt.title('3x3Filter')

plt.savefig("Challenge.png")