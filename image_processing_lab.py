# ============================================
# Image Processing for Scientific Applications
# Computational Science Laboratory
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------
# LOAD IMAGE
# --------------------------------------------

image = cv2.imread("potato_leaf.jpg")

if image is None:
    print("Error: potato_leaf.jpg not found.")
    exit()

# Convert BGR to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --------------------------------------------
# CONVERT TO GRAYSCALE
# --------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --------------------------------------------
# APPLY GAUSSIAN BLUR
# --------------------------------------------

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# --------------------------------------------
# EDGE DETECTION
# --------------------------------------------

edges = cv2.Canny(blur, 50, 150)

# --------------------------------------------
# IMAGE THRESHOLDING
# --------------------------------------------

_, threshold = cv2.threshold(
    gray,
    120,
    255,
    cv2.THRESH_BINARY
)

# --------------------------------------------
# CONTOUR DETECTION
# --------------------------------------------

contours, _ = cv2.findContours(
    threshold,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

image_contours = image_rgb.copy()

cv2.drawContours(
    image_contours,
    contours,
    -1,
    (255, 0, 0),
    2
)

# --------------------------------------------
# HISTOGRAM
# --------------------------------------------

hist = cv2.calcHist(
    [gray],
    [0],
    None,
    [256],
    [0, 256]
)

# --------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(edges, cmap="gray")
plt.title("Edge Detection")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(image_contours)
plt.title("Detected Objects")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.plot(hist)
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# --------------------------------------------
# IMAGE ANALYSIS RESULTS
# --------------------------------------------

print("--------------------------------------------")
print("IMAGE ANALYSIS RESULTS")
print("--------------------------------------------")
print("Image Width :", image.shape[1], "pixels")
print("Image Height:", image.shape[0], "pixels")
print("Number of Detected Objects:", len(contours))
