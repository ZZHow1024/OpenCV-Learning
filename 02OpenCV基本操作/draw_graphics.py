import numpy as np
import cv2 as cv
import zhplot
import matplotlib.pyplot as plt

image = np.zeros((512, 512, 3), np.uint8)

cv.line(image, (0, 0), (511, 511), (0, 0, 255), 3)
cv.rectangle(image, (0, 0), (300, 300), (255, 0, 0), 3)
cv.circle(image, (512 >> 1, 512 >> 1), 30, (0, 255, 0), -1)
cv.putText(image, 'OpenCV', (100, 200), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv.LINE_AA)

zhplot.matplotlib_chineseize()
plt.imshow(image[:, :, ::-1])
plt.title('OpenCV 绘制图形')
plt.show()