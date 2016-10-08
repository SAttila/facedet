import numpy as np
import cv2
from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'x1 y1 x2 y2')


lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")


# Returns a skin mask based on HSV values.
def detect_skin(frame):

    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=3)
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel2)

    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    return skinMask


def intersect_area(a, b):
    dx = min(a.x2, b.x2) - max(a.x1, b.x1)
    dy = min(a.y2, b.y2) - max(a.y1, b.y1)
    if (dx >= 0) and (dy >= 0):
        return dx * dy
