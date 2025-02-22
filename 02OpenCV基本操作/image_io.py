import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("images/image.jpg", cv.IMREAD_GRAYSCALE)

cv.imshow("image", image)
cv.waitKey(0)
cv.destroyAllWindows()

plt.imshow(image, cmap="gray")
plt.show()

cv.imwrite("images/image_gray.jpg", image)