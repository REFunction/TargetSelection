import cv2
import numpy as np
image = cv2.imread('../demo/hair2.jfif')
print(np.shape(image))
cv2.imwrite('../demo/hair2.png', image)
cv2.imshow('image', image)

cv2.waitKey(0)