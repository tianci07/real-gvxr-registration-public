import gvxrPython3 as gvxr
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from metrics import *
import skimage

def setXRayParameters(SOD, SDD):
    """Set up position of the X-ray source, object and the detector"""
    # Compute the source position in 3-D from the SOD
    gvxr.setSourcePosition(SOD,  0.0, 0.0, "cm")
    gvxr.setDetectorPosition(SOD - SDD, 0.0, 0.0, "cm")
    gvxr.usePointSource()

def setXRayEnvironment(aModelPath, aScale):
    """Set up initial environment for X-ray simulation"""
    gvxr.createWindow()
    gvxr.setWindowSize(512, 512)

    #gvxr.usePointSource()
    gvxr.setMonoChromatic(80, "keV", 1000)

    gvxr.setDetectorUpVector(0, 0, -1)

    img_width = 768
    img_height = 1024
    pixel_size = 0.5

    if aScale != 0:
        img_width /= 2**aScale
        img_height /= 2**aScale
        pixel_size *=2**aScale

    width = int(img_width)
    height = int(img_height)

    gvxr.setDetectorNumberOfPixels(width, height)
    gvxr.setDetectorPixelSize(pixel_size, pixel_size, "mm")

    setXRayParameters(90.0, 100.0)

    gvxr.loadSceneGraph(aModelPath, "m")
    node_label_set = []
    node_label_set.append('root')

    # The list is not empty
    while (len(node_label_set)):

        # Get the last node
        last_node = node_label_set[-1]

        # Initialise the material properties
        # print("Set ", label, "'s Hounsfield unit")
        # gvxr.setHU(label, 1000)
        Z = gvxr.getElementAtomicNumber("H")
        gvxr.setElement(last_node, gvxr.getElementName(Z))

        # Change the node colour to a random colour
        gvxr.setColour(last_node, random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1.0)

        # Remove it from the list
        node_label_set.pop()

        # Add its Children
        for i in range(gvxr.getNumberOfChildren(last_node)):
            node_label_set.append(gvxr.getChildLabel(last_node, i))

    gvxr.moveToCentre('root')
    gvxr.disableArtefactFiltering()
    gvxr.rotateNode('root', -90, 1, 0, 0)

def getLocalTransformationMatrixSet():
    # Parse the scenegraph
    matrix_set = {}

    node_label_set = []
    node_label_set.append('root')

    # The list is not empty
    while (len(node_label_set)):

        # Get the last node
        last_node = node_label_set[-1]

        # Get its local transformation
        matrix_set[last_node] = gvxr.getLocalTransformationMatrix(last_node)

        # Remove it from the list
        node_label_set.pop()

        # Add its Children
        for i in range(gvxr.getNumberOfChildren(last_node)):
            node_label_set.append(gvxr.getChildLabel(last_node, i))

    return matrix_set

def setLocalTransformationMatrixSet(aMatrixSet):
    # Restore the initial local transformation matrices
    for key in aMatrixSet:
        gvxr.setLocalTransformationMatrix(key, aMatrixSet[key])

def updateLocalTransformationMatrixSet(anAngle,  aFinger):
    """Rotate and/or rescale hand bones"""

    if aFinger == 'Root':

        gvxr.rotateNode('root', anAngle[0], 1, 0, 0)
        gvxr.rotateNode('root', anAngle[1], 0, 1, 0)
        gvxr.rotateNode('root', anAngle[2], 0, 0, 1)

    elif aFinger == 'Thumb':

        gvxr.rotateNode('node-Thu_Meta', anAngle[0], 1, 0, 0)
        gvxr.rotateNode('node-Thu_Meta', anAngle[1], 0, 1, 0)
        gvxr.rotateNode('node-Thu_Prox', anAngle[2], 1, 0, 0)

    elif aFinger == 'Little':

        gvxr.rotateNode('node-Lit_Prox', anAngle[0], 1, 0, 0)
        gvxr.rotateNode('node-Lit_Prox', anAngle[1], 0, 1, 0)
        gvxr.rotateNode('node-Lit_Midd', anAngle[2], 0, 1, 0)
        gvxr.rotateNode('node-Lit_Dist', anAngle[3], 0, 1, 0)

    elif aFinger == 'Ring':
        gvxr.rotateNode('node-Thi_Prox', anAngle[0], 1, 0, 0)
        gvxr.rotateNode('node-Thi_Prox', anAngle[1], 0, 1, 0)
        gvxr.rotateNode('node-Thi_Midd', anAngle[2], 0, 1, 0)
        gvxr.rotateNode('node-Thi_Dist', anAngle[3], 0, 1, 0)

    elif aFinger == 'Middle':
        gvxr.rotateNode('node-Mid_Prox', anAngle[0], 1, 0, 0)
        gvxr.rotateNode('node-Mid_Prox', anAngle[1], 0, 1, 0)
        gvxr.rotateNode('node-Mid_Midd', anAngle[2], 0, 1, 0)
        gvxr.rotateNode('node-Mid_Dist', anAngle[3], 0, 1, 0)

    elif aFinger == 'Index':
        gvxr.rotateNode('node-Ind_Prox', anAngle[0], 1, 0, 0)
        gvxr.rotateNode('node-Ind_Prox', anAngle[1], 0, 1, 0)
        gvxr.rotateNode('node-Ind_Midd', anAngle[2], 0, 1, 0)
        gvxr.rotateNode('node-Ind_Dist', anAngle[3], 0, 1, 0)

    elif aFinger == 'Rotate':

        gvxr.rotateNode('root', anAngle[0], 1, 0, 0)
        gvxr.rotateNode('root', anAngle[1], 0, 1, 0)
        gvxr.rotateNode('root', anAngle[2], 0, 0, 1)

        gvxr.rotateNode('node-Thu_Meta', anAngle[3], 1, 0, 0)
        gvxr.rotateNode('node-Thu_Meta', anAngle[4], 0, 1, 0)
        gvxr.rotateNode('node-Thu_Prox', anAngle[5], 1, 0, 0)

        gvxr.rotateNode('node-Ind_Prox', anAngle[6], 1, 0, 0)
        gvxr.rotateNode('node-Ind_Prox', anAngle[7], 0, 1, 0)
        gvxr.rotateNode('node-Ind_Midd', anAngle[8], 0, 1, 0)
        gvxr.rotateNode('node-Ind_Dist', anAngle[9], 0, 1, 0)

        gvxr.rotateNode('node-Mid_Prox', anAngle[10], 1, 0, 0)
        gvxr.rotateNode('node-Mid_Prox', anAngle[11], 0, 1, 0)
        gvxr.rotateNode('node-Mid_Midd', anAngle[12], 0, 1, 0)
        gvxr.rotateNode('node-Mid_Dist', anAngle[13], 0, 1, 0)

        gvxr.rotateNode('node-Thi_Prox', anAngle[14], 1, 0, 0)
        gvxr.rotateNode('node-Thi_Prox', anAngle[15], 0, 1, 0)
        gvxr.rotateNode('node-Thi_Midd', anAngle[16], 0, 1, 0)
        gvxr.rotateNode('node-Thi_Dist', anAngle[17], 0, 1, 0)

        gvxr.rotateNode('node-Lit_Prox', anAngle[18], 1, 0, 0)
        gvxr.rotateNode('node-Lit_Prox', anAngle[19], 0, 1, 0)
        gvxr.rotateNode('node-Lit_Midd', anAngle[20], 0, 1, 0)
        gvxr.rotateNode('node-Lit_Dist', anAngle[21], 0, 1, 0)

    elif aFinger == 'Rescale':

        # re-scale thumb
        gvxr.scaleNode('node-Thu_Prox', 1, 1, anAngle[0])
        gvxr.scaleNode('node-Thu_Dist', 1, 1, anAngle[0])

        # re-scale Index
        gvxr.scaleNode('node-Ind_Prox', 1, 1, anAngle[2])
        gvxr.scaleNode('node-Ind_Midd', 1, 1, anAngle[3])
        gvxr.scaleNode('node-Ind_Dist', 1, 1, anAngle[4])

        # re-scale Middle
        gvxr.scaleNode('node-Mid_Prox', 1, 1, anAngle[5])
        gvxr.scaleNode('node-Mid_Midd', 1, 1, anAngle[6])
        gvxr.scaleNode('node-Mid_Dist', 1, 1, anAngle[7])

        # re-scale Ring
        gvxr.scaleNode('node-Thi_Prox', 1, 1, anAngle[8])
        gvxr.scaleNode('node-Thi_Midd', 1, 1, anAngle[9])
        gvxr.scaleNode('node-Thi_Dist', 1, 1, anAngle[10])

        # re-scale Little
        gvxr.scaleNode('node-Lit_Prox', 1, 1, anAngle[11])
        gvxr.scaleNode('node-Lit_Midd', 1, 1, anAngle[12])
        gvxr.scaleNode('node-Lit_Dist', 1, 1, anAngle[13])

    elif aFinger == 'All':

        # Rotate bones
        gvxr.rotateNode('root', anAngle[0], 1, 0, 0)
        gvxr.rotateNode('root', anAngle[1], 0, 1, 0)
        gvxr.rotateNode('root', anAngle[2], 0, 0, 1)

        gvxr.rotateNode('node-Thu_Meta', anAngle[3], 1, 0, 0)
        gvxr.rotateNode('node-Thu_Meta', anAngle[4], 0, 1, 0)
        gvxr.rotateNode('node-Thu_Prox', anAngle[5], 1, 0, 0)

        gvxr.rotateNode('node-Ind_Prox', anAngle[6], 1, 0, 0)
        gvxr.rotateNode('node-Ind_Prox', anAngle[7], 0, 1, 0)
        gvxr.rotateNode('node-Ind_Midd', anAngle[8], 0, 1, 0)
        gvxr.rotateNode('node-Ind_Dist', anAngle[9], 0, 1, 0)

        gvxr.rotateNode('node-Mid_Prox', anAngle[10], 1, 0, 0)
        gvxr.rotateNode('node-Mid_Prox', anAngle[11], 0, 1, 0)
        gvxr.rotateNode('node-Mid_Midd', anAngle[12], 0, 1, 0)
        gvxr.rotateNode('node-Mid_Dist', anAngle[13], 0, 1, 0)

        gvxr.rotateNode('node-Thi_Prox', anAngle[14], 1, 0, 0)
        gvxr.rotateNode('node-Thi_Prox', anAngle[15], 0, 1, 0)
        gvxr.rotateNode('node-Thi_Midd', anAngle[16], 0, 1, 0)
        gvxr.rotateNode('node-Thi_Dist', anAngle[17], 0, 1, 0)

        gvxr.rotateNode('node-Lit_Prox', anAngle[18], 1, 0, 0)
        gvxr.rotateNode('node-Lit_Prox', anAngle[19], 0, 1, 0)
        gvxr.rotateNode('node-Lit_Midd', anAngle[20], 0, 1, 0)
        gvxr.rotateNode('node-Lit_Dist', anAngle[21], 0, 1, 0)

        # Rescale bones
        # Thumb
        gvxr.scaleNode('node-Thu_Prox', 1, 1, anAngle[22])
        gvxr.scaleNode('node-Thu_Dist', 1, 1, anAngle[23])

        # re-scale Index
        gvxr.scaleNode('node-Ind_Prox', 1, 1, anAngle[24])
        gvxr.scaleNode('node-Ind_Midd', 1, 1, anAngle[25])
        gvxr.scaleNode('node-Ind_Dist', 1, 1, anAngle[26])

        # re-scale Middle
        gvxr.scaleNode('node-Mid_Prox', 1, 1, anAngle[27])
        gvxr.scaleNode('node-Mid_Midd', 1, 1, anAngle[28])
        gvxr.scaleNode('node-Mid_Dist', 1, 1, anAngle[29])

        # re-scale Ring
        gvxr.scaleNode('node-Thi_Prox', 1, 1, anAngle[30])
        gvxr.scaleNode('node-Thi_Midd', 1, 1, anAngle[31])
        gvxr.scaleNode('node-Thi_Dist', 1, 1, anAngle[32])

        # re-scale Little
        gvxr.scaleNode('node-Lit_Prox', 1, 1, anAngle[33])
        gvxr.scaleNode('node-Lit_Midd', 1, 1, anAngle[34])
        gvxr.scaleNode('node-Lit_Dist', 1, 1, anAngle[35])

def boneRotation(anAngle, aFinger):
    """
    @Params:
        anAngle: list of rotating angles.

        aFinger: choice of rotation and/or rescaling bones. "Rotate", "Rescale",
                "Root", "Thumb", "Index", "Middle", "Ring", "Little",
                "None" or "All"
    """

    matrix_set = getLocalTransformationMatrixSet()

    if aFinger != 'None':
        updateLocalTransformationMatrixSet(anAngle, aFinger)

    x_ray_image = gvxr.computeXRayImage()
    image = np.array(x_ray_image)
    image = (image-image.mean())/image.std()

    setLocalTransformationMatrixSet(matrix_set)

    return image

def getTargetImage(aTarget, aScale, log=True):
    """Get a ground truth image"""

    # read target image
    target_image = cv2.imread("./"+aTarget, 0)

    #set nan and inf to zero
    target_image[np.isnan(target_image)]=0.
    target_image[np.isinf(target_image)] = 0.
    target_image=np.float32(target_image)

    if aScale != 0:
        # image pyramids - down scale
        for p in range(aScale):
            target_image = cv2.pyrDown(target_image)

    if log:
        target_image = np.log(target_image)

    # zero-mean normalisation
    target_image = (target_image-target_image.mean())/target_image.std()

    return target_image

def computePredictedImage(aPrediction):
    """Compute predicted image from predicted parameters
    @Parameters:
        aPrediction: a set of parameters for determining predicted images
    """
    if aPrediction.ndim >= 2:
        aPrediction = aPrediction[0,:]

    number_of_angles = 36
    number_of_distances = 2

    SOD = aPrediction[0]*aPrediction[1]
    SDD = aPrediction[1]
    SOD = np.float(SOD.item())
    SDD = np.float(SDD.item())

    setXRayParameters(SOD, SDD)

    best_angle = []
    for a in range(number_of_angles):
        best_angle.append(aPrediction[a+number_of_distances])

    pred_image = boneRotation(best_angle, 'All')

    return pred_image
