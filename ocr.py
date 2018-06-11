from PIL import Image
import pytesseract
import cv2
import numpy as np
import time

image = cv2.imread("./1.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

invert = cv2.bitwise_not(threshold)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
dilate = cv2.dilate(invert, kernel, iterations = 1)

height, width = dilate.shape

mask = np.zeros((height + 2, width + 2), np.uint8)

flood = dilate.copy()

maxPoint = (-1, -1)
maxArea = -1

for y in range(0, height):
    for x in range(0, width):
        if flood[y][x] >= 128:

            mask[:] = 0

            area = cv2.floodFill(flood, mask, (x, y), 64)[0]

            if area > maxArea:
                maxArea = area
                maxPoint = (x, y)

mask[:] = 0

cv2.floodFill(flood, mask, maxPoint, 255)

boxes = flood.copy()

for y in range(0, height):
    for x in range(0, width):
        if boxes[y][x] == 64:
            boxes[y][x] = 0

fakeImage, contours, hierarchy = cv2.findContours(boxes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contoursImage = np.zeros((height, width, 1), np.uint8)
contoursImage = cv2.drawContours(contoursImage, contours, -1, (255, 255, 255), 5)

linesImage = np.zeros((height, width, 3), np.uint8)
edges = cv2.Canny(contoursImage, 75, 150, apertureSize = 3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, 0, 0)


corners = cv2.cornerHarris(np.float32(contoursImage), 2, 3, 0.04)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(linesImage, (x1, y1), (x2, y2), (255, 0, 255), 2)

cv2.imwrite("2.jpg", gray)
cv2.imwrite("3.jpg", blur)
cv2.imwrite("4.jpg", threshold)
cv2.imwrite("5.jpg", invert)
cv2.imwrite("6.jpg", dilate)
cv2.imwrite("7.jpg", flood)
cv2.imwrite("8.jpg", boxes)
cv2.imwrite("9.jpg", contoursImage)
cv2.imwrite("10.jpg", edges)
cv2.imwrite("11.jpg", linesImage)
cv2.imwrite("12.jpg", corners)

#print(pytesseract.image_to_string(image).encode("utf-8"))
