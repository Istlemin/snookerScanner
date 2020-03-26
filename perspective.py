import cv2, math
from hillClimb import *
import numpy as np
from matplotlib import pyplot as plt

def mat_set(m,x,y,v):
    m[x-1][y-1]=v

def mat_perspective(angle, ratio,near, far):
    to_return = np.zeros((4,4))
    tan_half_angle = math.tan(angle / 2)

    mat_set(to_return, 1, 1, 1.0 / (ratio * tan_half_angle))
    mat_set(to_return, 2, 2, 1.0 / (tan_half_angle))
    mat_set(to_return, 3, 3, -(far + near) / (far - near))
    mat_set(to_return, 4, 3, -1.0)
    mat_set(to_return, 3, 4, -(2.0 * far * near) / (far - near))
    return to_return

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac),0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab),0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc,0],
                     [0,0,0,1]])

def project(point,matrix,fov):
    if fov==0:
        return np.array([1e18,1e18])

    projectionMatrix = mat_perspective(fov,1,0.1,100)

    ort = matrix.dot(np.concatenate([point,[1]]))
    xp,yp,zp,wp = projectionMatrix.dot(ort)

    return np.array([xp/wp,yp/wp])

def drawPerspectiveLine(img,p1,p2,matrix,color):
    global width,height
    pos1 = project(p1,matrix)
    pos2 = project(p2,matrix)

    pos1*=200
    pos2*=200
    pos1+=np.float64([width/2,height/2])
    pos2+=np.float64([width/2,height/2])

    cv2.line(img, tuple(pos1.astype(int)), tuple(pos2.astype(int)), color,1)

def drawLine(img,pos1,pos2,color):
    global width,height

    p1 = pos1+np.int64([width/2,height/2])
    p2 = pos2+np.int64([width/2,height/2])


    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), color,2)


def drawPerspectiveRect(img,rect,matrix,color):
    for i in range(4):
        drawPerspectiveLine(img,rect[i],rect[(i+1)%4],matrix,color)

def drawRect(img,rect,color):
    for i in range(4):
        drawLine(img,rect[i],rect[(i+1)%4],color)

def transformPoint(point,cx,cy,cz,pitch,roll,yaw,fov,scale):
    cameraMatrix = np.array([
        [1,0,0,-cx],
        [0,1,0,-cy],
        [0,0,1,-cz],
        [0,0,0,1]
    ])
    rotationMatrix = rotation_matrix([1,0,0],-pitch)
    cameraMatrix = rotationMatrix.dot(cameraMatrix)

    return project(point,cameraMatrix,fov)*scale

def transformRect(rect,cx,cy,cz,pitch,roll,yaw,fov,scale):

    cameraMatrix = np.array([
        [1,0,0,-cx],
        [0,1,0,-cy],
        [0,0,1,-cz],
        [0,0,0,1]
    ])
    rotationMatrix = rotation_matrix([1,0,0],-pitch)
    cameraMatrix = rotationMatrix.dot(cameraMatrix)

    tramsformedRect = np.zeros((4,2))
    for i in range(len(rect)):
        newPos = project(rect[i],cameraMatrix,fov)
        tramsformedRect[i][0] = newPos[0]*scale
        tramsformedRect[i][1] = newPos[1]*scale
    return tramsformedRect

def rectDistance(rectA,rectB):
    ans = 0
    for p1,p2 in zip(rectA,rectB):
        ans += np.linalg.norm(p1-p2)**2
    return ans

width=0
height=0
def getBallPositions3d(img,tableCornersImg,ballPositionsImg):
    global width, height
    width = img.shape[1]
    height = img.shape[0]
    center = np.array([width/2,height/2])

    tableCorners3d = np.array([[-100,-200,10],[100,-200,10],[100,200,10],[-100,200,10]])

    fov = 0.03
    angle = 1.18
    scale = 200
    cz = 500
    cy = 1000

    print("Hill Climbing to find camera position...")
    angle,cz,scale,cy = hillClimb([[0.5,1.5],[100,1000],[40,600],[400,6000]],
    lambda params: rectDistance(transformRect(tableCorners3d,0,params[3],params[1],-params[0],0,0,fov,params[2]),tableCornersImg))    


    debug = img.copy()

    #print(angle,cy,cz,scale)
    drawRect(debug,transformRect(tableCorners3d,0,cy,cz,-angle,0,0,fov,scale),(255,0,0))

    ballPositions3d = []

    ballZ = 8
    for ballImg in ballPositionsImg:
        bx,by = hillClimb([[-100,100],[-200,200]],lambda pos: np.linalg.norm(transformPoint([pos[0],pos[1],ballZ],0,cy,cz,-angle,0,0,fov,scale)-(ballImg-center)))
        #print(bx,by)
        ballPositions3d.append(np.array([bx,by]))
        cv2.circle(debug,tuple((transformPoint([bx,by,ballZ],0,cy,cz,-angle,0,0,fov,scale)+center).astype(int)),3,(0,0,255),3)
    
    cv2.imwrite("debug/debug8.png",debug)
    return ballPositions3d


#bottomcorners = np.array([[-100,-200,-10],[100,-200,-10],[100,200,-10],[-100,200,-10]])

