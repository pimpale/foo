#! /bin/python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
img = cv.imread(sys.argv[1],0)
edges = cv.Canny(img,100,200)
while True:
    cv.imshow('edges',edges)

