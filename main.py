import cv2
import numpy as np
from matplotlib import pyplot as plt
from imageRecognition import *
from perspective import *

SOURCE_IMAGE_PATH = "sampleImages/snooker.png"

def getRenderedImage(ballPositions,ballColors):
    img = cv2.imread('snookerTable.png')
    width = img.shape[1]
    height = img.shape[0]

    ballSize = 5.25/366*width/2

    SNOOKER_COLORS = [
        [[20,20,20],[20,20,20]],
        [[149,168,255],[149,168,255]],
        [[160,92,17],[160,92,17]],
        [[0,85,153],[0,85,153]],
        [[30,90,0],[30,90,0]],
        [[0,224,255],[0,224,255]],
        [[208,251,248],[208,251,248]],
        [[40,50,160],[0,0,200]]
    ]

    scale = (width/2-35)/200

    for i,(bx,by) in enumerate(ballPositions):
        renderX = int(-by*scale+width/2)
        renderY = int(bx*scale+height/2)

        alpha = 1.3
        beta = -15
        ballColors[i] = np.clip(alpha*np.array(ballColors[i]) + beta, 0, 255)

        colorDistances = np.array(list(map(lambda c: np.linalg.norm(np.array(c[0])-np.array(ballColors[i])), SNOOKER_COLORS)))
        ballColors[i] = SNOOKER_COLORS[np.argmin(colorDistances)][1]
        cv2.circle(img, (renderX,renderY), int(ballSize), ballColors[i], -1)
    return img


src=cv2.imread(SOURCE_IMAGE_PATH)

tableCornersInSrc = getTableRect(src)

ballPositionsInSrc,ballColors = findBalls(src,tableCornersInSrc)

for i in range(4):
    tableCornersInSrc[i] -= np.array([src.shape[1]/2,src.shape[0]/2])

ballPositionsIn3d = getBallPositions3d(src,tableCornersInSrc,ballPositionsInSrc)

render = src
render = getRenderedImage(ballPositionsIn3d,ballColors)

render = cv2.resize(render,(1400,700))

cv2.imwrite("render.png",render)
cv2.imwrite("debug/src.png",src)

print("Done! Output written to render.png")
