import cv2
import numpy as np
from matplotlib import pyplot as plt

def getLineEquation(p1,p2):
    k = (p1[1]-p2[1])/(p1[0]-p2[0])
    m = p1[1]-k*p1[0]
    return (k,m)

def intersectLineEq(k1,m1,k2,m2):
    x = (m1-m2)/(k2-k1)
    y = k1*x+m1
    return np.array([x,y])

def averageColorOfLine(img,p1,p2):
    colorSum = np.array([0,0,0])
    samples = 100
    for i in range(samples):
        samplePoint = (i/(samples-1))*p2+(1-i/(samples-1))*p1
        samplePoint = samplePoint.astype(int)
        x = samplePoint[0]
        y = samplePoint[1]
        x = min(img.shape[1]-1,x)
        y = min(img.shape[0]-1,y)
        colorSum += np.array(img[y][x])

    return colorSum/samples

def isGreen(color):
    if np.linalg.norm(color)==0:
        return False
    normalizedColor = color/np.linalg.norm(color)
    return normalizedColor[1]>=0.73

def isBrown(color):
    if np.linalg.norm(color)==0:
        return False
    normalizedColor = color/np.linalg.norm(color)
    return not isGreen(color) and np.linalg.norm(normalizedColor-np.array([ 0.54,0.47,0.69]))<=0.30

def isTableEdge(img,p1,p2):
    translateVector = np.array((p1[1]-p2[1],p2[0]-p1[0]))
    translateVector = translateVector/np.linalg.norm(translateVector)

    dist = 5
    color1 = averageColorOfLine(img,p1+translateVector*dist,p2+translateVector*dist)
    color2 = averageColorOfLine(img,p1-translateVector*dist,p2-translateVector*dist)

    if isBrown(color1) and isGreen(color2):
        return True
    if isBrown(color2) and isGreen(color1):
        return True
    return False

def whichTableEdge(img,p1,p2):
    if abs(p1[0]-p2[0])>abs(p1[1]-p2[1]):#top or bottom
        if p1[1]+p2[1]<img.shape[0]:#top
            return 0
        else:#bottom
            return 2
    else:#left or right
        if p1[0]+p2[0]<img.shape[1]:#left
            return 3
        else:#right
            return 1

def getTableRectLines(img):
    global drawImage
    edges = cv2.Canny(img,50,100)

    cv2.imwrite("debug/debug1.png",edges)
    #cv2.imshow('image',edges)
    #cv2.waitKey(0)
    #cv2.destroyWindow()
    minLineLength = 160
    maxLineGap = 30
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,100, minLineLength, maxLineGap)

    rectLines = [[],[],[],[]]

    debug2 = img.copy()
    for line in lines:
        (x1, y1, x2, y2) = line[0]
        p1 = np.array((x1,y1))
        p2 = np.array((x2,y2))
        cv2.line(debug2, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 0, 0), 2)
        if isTableEdge(img,p1,p2):
            rectLines[whichTableEdge(img,p1,p2)].append([p1,p2])
    cv2.imwrite("debug/debug2.png",debug2)
    return rectLines

def getTableRect(img):
    global drawImage
    print("Finding table edges...")
    rectLines = getTableRectLines(img)


    debug3 = img.copy()

    rectLineEqs = []
    for lines in rectLines:
        if len(lines)==0:
            continue
        #assert(len(lines)!=0)

        #kSum = 0
        #mSum = 0
        k = 0
        m = 0
        d = 0
        for l in lines:
            cv2.line(debug3, tuple(l[0].astype(int)), tuple(l[1].astype(int)), (0, 0, 255), 2)
            k1,m1 = getLineEquation(l[0],l[1])
            if np.linalg.norm(l[0]-l[1])>d:
                d = np.linalg.norm(l[0]-l[1])
                k = k1
                m = m1
            #kSum+=k1
            #mSum+=m1
        #rectLineEqs.append([kSum/len(lines),mSum/len(lines)])
        rectLineEqs.append([k,m])

    cv2.imwrite("debug/debug3.png",debug3)
    for lines in rectLines:
        if len(lines)==0:
            return

    tableCorners = []
    for i in range(4):
        k1,m1 = rectLineEqs[i]
        k2,m2 = rectLineEqs[(i+3)%4]
        tableCorners.append(intersectLineEq(k1,m1,k2,m2))
    
    debug7 = img.copy()
    for i in range(4):
        cv2.line(debug7, tuple(tableCorners[i].astype(int)), tuple(tableCorners[(i+1)%4].astype(int)), (0, 255, 0), 2)
    cv2.imwrite("debug/debug7.png",debug7)
    return np.array(tableCorners)

def getTableColor(img,tableCorners):
    x1,y1 = tableCorners[0]
    x2 = tableCorners[1][0]
    y2 = tableCorners[3][1]
    res = cv2.mean(img[int(y1):int(y2*1/3+y1*2/3),int(x1):int(x2)])
    return np.array([res[0],res[1],res[2]])

def findBalls(img,tableCorners):
    print("Finding the balls...")
    green = getTableColor(img,tableCorners)
    #print(np.array(tableCorners))
    maskCorners = np.array(tableCorners)
    shrinkage = 50
    maskCorners[0]+=np.array([shrinkage*0.6,shrinkage*0.6])
    maskCorners[1]+=np.array([-shrinkage*0.6,shrinkage*0.6])
    maskCorners[2]+=np.array([-shrinkage,-shrinkage*0.4])
    maskCorners[3]+=np.array([shrinkage,-shrinkage*0.4])
    tableMask = cv2.fillConvexPoly(np.zeros_like(img),np.array(maskCorners).astype(int) , (1,1,1))
    
    edges = cv2.Canny(img,100,5)

    edges*=tableMask[:,:,0]

    cv2.imwrite("debug/debug5.png",edges)

    detected_circles = cv2.HoughCircles(edges,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 20, 
               param2 = 13, minRadius = 8, maxRadius = 17) 
    
    edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    
    ballsPositions = []
    ballColors = []

    if detected_circles is not None: 
        detected_circles = np.uint16(np.around(detected_circles)) 

        debug6 = img.copy()
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
            cv2.circle(debug6, (a, b), r, (0, 255, 0), 2) 
            #cv2.circle(img, (a, int(b+r*0.7)), 1, (0, 0, 255), 3)
            ballsPositions.append(np.array([a, int(b)]))

            x1 =  int(a-2**0.5*r/2)
            x2 =  int(a+2**0.5*r/2)
            y1 =  int(b-2**0.5*r/2)
            y2 =  int(b+2**0.5*r/2)

            #print(x1,x2,y1,y2)
            colorRect = img[y1:y2,x1:x2]
            colorSum = np.array([0,0,0])
            for i in range(colorRect.shape[0]):
                for j in range(colorRect.shape[1]):
                    colorSum+=colorRect[i][j].astype(int)

            #print(colorSum)

            width = int(colorRect.shape[0])
            height = int(colorRect.shape[1])
            avgColor = [int(colorSum[0]/width/height),int(colorSum[1]/width/height),int(colorSum[2]/width/height)]
            ballColors.append(avgColor)
        
        cv2.imwrite("debug/debug6.png",debug6)
    #cv2.imshow('image',edges)
    #cv2.waitKey(0)
    #cv2.destroyWindow()

    return ballsPositions,ballColors
