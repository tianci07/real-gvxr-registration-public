import gvxrPython3 as gvxr
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from metrics import *

def setXRayParameters(SOD, SDD):
    """Set up position of the X-ray source, object and the detector"""
    # Compute the source position in 3-D from the SOD
    gvxr.setSourcePosition(SOD,  0.0, 0.0, "cm");
    gvxr.setDetectorPosition(SOD - SDD, 0.0, 0.0, "cm");
    gvxr.usePointSource();

def setXRayEnvironment(aScale):
    """Set up initial environment for X-ray simulation"""
    gvxr.createWindow();
    gvxr.setWindowSize(512, 512);

    #gvxr.usePointSource();
    gvxr.setMonoChromatic(80, "keV", 1000);

    gvxr.setDetectorUpVector(0, 0, -1);

    img_width = 768;
    img_height = 1024;
    pixel_size = 0.5;

    if aScale != 0:
        img_width /= 2**aScale;
        img_height /= 2**aScale;
        pixel_size *=2**aScale;

    width = int(img_width);
    height = int(img_height);
    gvxr.setDetectorNumberOfPixels(width, height);
    gvxr.setDetectorPixelSize(pixel_size, pixel_size, "mm");

    setXRayParameters(10.0, 100.0);

    gvxr.loadSceneGraph("./hand.dae", "m");
    node_label_set = [];
    node_label_set.append('root');

    # The list is not empty
    while (len(node_label_set)):

        # Get the last node
        last_node = node_label_set[-1];

        # Initialise the material properties
        # print("Set ", label, "'s Hounsfield unit");
        # gvxr.setHU(label, 1000)
        Z = gvxr.getElementAtomicNumber("H");
        gvxr.setElement(last_node, gvxr.getElementName(Z));

        # Change the node colour to a random colour
        gvxr.setColour(last_node, random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1.0);

        # Remove it from the list
        node_label_set.pop();

        # Add its Children
        for i in range(gvxr.getNumberOfChildren(last_node)):
            node_label_set.append(gvxr.getChildLabel(last_node, i));

    gvxr.moveToCentre('root');
    gvxr.disableArtefactFiltering();
    gvxr.rotateNode('root', -90, 1, 0, 0);

def getLocalTransformationMatrixSet():
    # Parse the scenegraph
    matrix_set = {};

    node_label_set = [];
    node_label_set.append('root');

    # The list is not empty
    while (len(node_label_set)):

        # Get the last node
        last_node = node_label_set[-1];

        # Get its local transformation
        matrix_set[last_node] = gvxr.getLocalTransformationMatrix(last_node);

        # Remove it from the list
        node_label_set.pop();

        # Add its Children
        for i in range(gvxr.getNumberOfChildren(last_node)):
            node_label_set.append(gvxr.getChildLabel(last_node, i));

    return matrix_set;

def setLocalTransformationMatrixSet(aMatrixSet):
    # Restore the initial local transformation matrices
    for key in aMatrixSet:
        gvxr.setLocalTransformationMatrix(key, aMatrixSet[key]);

def updateLocalTransformationMatrixSet(anAngle,  aFinger):
    """Rotate and/or rescale hand bones"""

    if aFinger == 'Root':

        gvxr.rotateNode('root', anAngle[0], 1, 0, 0);
        gvxr.rotateNode('root', anAngle[1], 0, 1, 0);
        gvxr.rotateNode('root', anAngle[2], 0, 0, 1);

    elif aFinger == 'Thumb':

        gvxr.rotateNode('node-Thu_Meta', anAngle[0], 1, 0, 0);
        gvxr.rotateNode('node-Thu_Meta', anAngle[1], 0, 1, 0);
        gvxr.rotateNode('node-Thu_Prox', anAngle[2], 1, 0, 0);

    elif aFinger == 'Little':

        gvxr.rotateNode('node-Lit_Prox', anAngle[0], 1, 0, 0);
        gvxr.rotateNode('node-Lit_Prox', anAngle[1], 0, 1, 0);
        gvxr.rotateNode('node-Lit_Midd', anAngle[2], 0, 1, 0);
        gvxr.rotateNode('node-Lit_Dist', anAngle[3], 0, 1, 0);

    elif aFinger == 'Ring':
        gvxr.rotateNode('node-Thi_Prox', anAngle[0], 1, 0, 0);
        gvxr.rotateNode('node-Thi_Prox', anAngle[1], 0, 1, 0);
        gvxr.rotateNode('node-Thi_Midd', anAngle[2], 0, 1, 0);
        gvxr.rotateNode('node-Thi_Dist', anAngle[3], 0, 1, 0);

    elif aFinger == 'Middle':
        gvxr.rotateNode('node-Mid_Prox', anAngle[0], 1, 0, 0);
        gvxr.rotateNode('node-Mid_Prox', anAngle[1], 0, 1, 0);
        gvxr.rotateNode('node-Mid_Midd', anAngle[2], 0, 1, 0);
        gvxr.rotateNode('node-Mid_Dist', anAngle[3], 0, 1, 0);

    elif aFinger == 'Index':
        gvxr.rotateNode('node-Ind_Prox', anAngle[0], 1, 0, 0);
        gvxr.rotateNode('node-Ind_Prox', anAngle[1], 0, 1, 0);
        gvxr.rotateNode('node-Ind_Midd', anAngle[2], 0, 1, 0);
        gvxr.rotateNode('node-Ind_Dist', anAngle[3], 0, 1, 0);

    elif aFinger == 'Rotate':

        gvxr.rotateNode('root', anAngle[0], 1, 0, 0);
        gvxr.rotateNode('root', anAngle[1], 0, 1, 0);
        gvxr.rotateNode('root', anAngle[2], 0, 0, 1);

        gvxr.rotateNode('node-Thu_Meta', anAngle[3], 1, 0, 0);
        gvxr.rotateNode('node-Thu_Meta', anAngle[4], 0, 1, 0);
        gvxr.rotateNode('node-Thu_Prox', anAngle[5], 1, 0, 0);

        gvxr.rotateNode('node-Ind_Prox', anAngle[6], 1, 0, 0);
        gvxr.rotateNode('node-Ind_Prox', anAngle[7], 0, 1, 0);
        gvxr.rotateNode('node-Ind_Midd', anAngle[8], 0, 1, 0);
        gvxr.rotateNode('node-Ind_Dist', anAngle[9], 0, 1, 0);

        gvxr.rotateNode('node-Mid_Prox', anAngle[10], 1, 0, 0);
        gvxr.rotateNode('node-Mid_Prox', anAngle[11], 0, 1, 0);
        gvxr.rotateNode('node-Mid_Midd', anAngle[12], 0, 1, 0);
        gvxr.rotateNode('node-Mid_Dist', anAngle[13], 0, 1, 0);

        gvxr.rotateNode('node-Thi_Prox', anAngle[14], 1, 0, 0);
        gvxr.rotateNode('node-Thi_Prox', anAngle[15], 0, 1, 0);
        gvxr.rotateNode('node-Thi_Midd', anAngle[16], 0, 1, 0);
        gvxr.rotateNode('node-Thi_Dist', anAngle[17], 0, 1, 0);

        gvxr.rotateNode('node-Lit_Prox', anAngle[18], 1, 0, 0);
        gvxr.rotateNode('node-Lit_Prox', anAngle[19], 0, 1, 0);
        gvxr.rotateNode('node-Lit_Midd', anAngle[20], 0, 1, 0);
        gvxr.rotateNode('node-Lit_Dist', anAngle[21], 0, 1, 0);

    elif aFinger == 'Rescale':

        # re-scale thumb
        gvxr.scaleNode('node-Thu_Prox', 1, 1, anAngle[0], 'mm');
        gvxr.scaleNode('node-Thu_Dist', 1, 1, anAngle[0], 'mm');

        # re-scale Index
        gvxr.scaleNode('node-Ind_Prox', 1, 1, anAngle[2], 'mm');
        gvxr.scaleNode('node-Ind_Midd', 1, 1, anAngle[3], 'mm');
        gvxr.scaleNode('node-Ind_Dist', 1, 1, anAngle[4], 'mm');

        # re-scale Middle
        gvxr.scaleNode('node-Mid_Prox', 1, 1, anAngle[5], 'mm');
        gvxr.scaleNode('node-Mid_Midd', 1, 1, anAngle[6], 'mm');
        gvxr.scaleNode('node-Mid_Dist', 1, 1, anAngle[7], 'mm');

        # re-scale Ring
        gvxr.scaleNode('node-Thi_Prox', 1, 1, anAngle[8], 'mm');
        gvxr.scaleNode('node-Thi_Midd', 1, 1, anAngle[9], 'mm');
        gvxr.scaleNode('node-Thi_Dist', 1, 1, anAngle[10], 'mm');

        # re-scale Little
        gvxr.scaleNode('node-Lit_Prox', 1, 1, anAngle[11], 'mm');
        gvxr.scaleNode('node-Lit_Midd', 1, 1, anAngle[12], 'mm');
        gvxr.scaleNode('node-Lit_Dist', 1, 1, anAngle[13], 'mm');

    elif aFinger == 'All':

        # Rotate bones
        gvxr.rotateNode('root', anAngle[0], 1, 0, 0);
        gvxr.rotateNode('root', anAngle[1], 0, 1, 0);
        gvxr.rotateNode('root', anAngle[2], 0, 0, 1);

        gvxr.rotateNode('node-Thu_Meta', anAngle[3], 1, 0, 0);
        gvxr.rotateNode('node-Thu_Meta', anAngle[4], 0, 1, 0);
        gvxr.rotateNode('node-Thu_Prox', anAngle[5], 1, 0, 0);

        gvxr.rotateNode('node-Ind_Prox', anAngle[6], 1, 0, 0);
        gvxr.rotateNode('node-Ind_Prox', anAngle[7], 0, 1, 0);
        gvxr.rotateNode('node-Ind_Midd', anAngle[8], 0, 1, 0);
        gvxr.rotateNode('node-Ind_Dist', anAngle[9], 0, 1, 0);

        gvxr.rotateNode('node-Mid_Prox', anAngle[10], 1, 0, 0);
        gvxr.rotateNode('node-Mid_Prox', anAngle[11], 0, 1, 0);
        gvxr.rotateNode('node-Mid_Midd', anAngle[12], 0, 1, 0);
        gvxr.rotateNode('node-Mid_Dist', anAngle[13], 0, 1, 0);

        gvxr.rotateNode('node-Thi_Prox', anAngle[14], 1, 0, 0);
        gvxr.rotateNode('node-Thi_Prox', anAngle[15], 0, 1, 0);
        gvxr.rotateNode('node-Thi_Midd', anAngle[16], 0, 1, 0);
        gvxr.rotateNode('node-Thi_Dist', anAngle[17], 0, 1, 0);

        gvxr.rotateNode('node-Lit_Prox', anAngle[18], 1, 0, 0);
        gvxr.rotateNode('node-Lit_Prox', anAngle[19], 0, 1, 0);
        gvxr.rotateNode('node-Lit_Midd', anAngle[20], 0, 1, 0);
        gvxr.rotateNode('node-Lit_Dist', anAngle[21], 0, 1, 0);

        # Rescale bones
        # Thumb
        gvxr.scaleNode('node-Thu_Prox', 1, 1, anAngle[22], 'mm');
        gvxr.scaleNode('node-Thu_Dist', 1, 1, anAngle[23], 'mm');

        # re-scale Index
        gvxr.scaleNode('node-Ind_Prox', 1, 1, anAngle[24], 'mm');
        gvxr.scaleNode('node-Ind_Midd', 1, 1, anAngle[25], 'mm');
        gvxr.scaleNode('node-Ind_Dist', 1, 1, anAngle[26], 'mm');

        # re-scale Middle
        gvxr.scaleNode('node-Mid_Prox', 1, 1, anAngle[27], 'mm');
        gvxr.scaleNode('node-Mid_Midd', 1, 1, anAngle[28], 'mm');
        gvxr.scaleNode('node-Mid_Dist', 1, 1, anAngle[29], 'mm');

        # re-scale Ring
        gvxr.scaleNode('node-Thi_Prox', 1, 1, anAngle[30], 'mm');
        gvxr.scaleNode('node-Thi_Midd', 1, 1, anAngle[31], 'mm');
        gvxr.scaleNode('node-Thi_Dist', 1, 1, anAngle[32], 'mm');

        # re-scale Little
        gvxr.scaleNode('node-Lit_Prox', 1, 1, anAngle[33], 'mm');
        gvxr.scaleNode('node-Lit_Midd', 1, 1, anAngle[34], 'mm');
        gvxr.scaleNode('node-Lit_Dist', 1, 1, anAngle[35], 'mm');

def computeAverageHand():
    """
    Rescale hand bones to average size according to the hand
    measurements from 20 patients
    """
    # re-scale Thumb
    gvxr.scaleNode('node-Thu_Prox', 1, 1, 1.086, 'mm');
    gvxr.scaleNode('node-Thu_Dist', 1, 1, 0.897, 'mm');

    # re-scale Index
    gvxr.scaleNode('node-Ind_Prox', 1, 1, 0.969, 'mm');
    gvxr.scaleNode('node-Ind_Midd', 1, 1, 1.065, 'mm');
    gvxr.scaleNode('node-Ind_Dist', 1, 1, 1.141, 'mm');

    # re-scale Middle
    gvxr.scaleNode('node-Mid_Prox', 1, 1, 0.962, 'mm');
    gvxr.scaleNode('node-Mid_Midd', 1, 1, 1.080, 'mm');
    gvxr.scaleNode('node-Mid_Dist', 1, 1, 1.053, 'mm');

    # re-scale Ring
    gvxr.scaleNode('node-Thi_Prox', 1, 1, 1.017, 'mm');
    gvxr.scaleNode('node-Thi_Midd', 1, 1, 1.084, 'mm');
    gvxr.scaleNode('node-Thi_Dist', 1, 1, 1.056, 'mm');

    # re-scale Little
    gvxr.scaleNode('node-Lit_Prox', 1, 1, 1.034, 'mm');
    gvxr.scaleNode('node-Lit_Midd', 1, 1, 1.126, 'mm');
    gvxr.scaleNode('node-Lit_Dist', 1, 1, 1.070, 'mm');

def boneRotation(anAngle, aFinger):
    """
    @Params:
        anAngle: list of rotating angles.

        aFinger: choice of rotation and/or rescaling bones. "Rotate", "Rescale",
                "Root", "Thumb", "Index", "Middle", "Ring", "Little",
                "None" or "All"
    """

    matrix_set = getLocalTransformationMatrixSet();

    if aFinger != 'None':
        updateLocalTransformationMatrixSet(anAngle, aFinger);

    x_ray_image = gvxr.computeXRayImage();
    image = np.array(x_ray_image);
    image = (image-image.mean())/image.std();

    setLocalTransformationMatrixSet(matrix_set);

    return image

def computePredictedImage(aPrediction):
    """Compute predicted image from predicted parameters
    @Parameters:
        aPrediction: a set of parameters for determining predicted images
    """
    number_of_angles = 36;
    number_of_distances = 2;

    SOD = aPrediction[0]*aPrediction[1];
    SDD = aPrediction[1];
    SOD = np.float(SOD.item());
    SDD = np.float(SDD.item());

    setXRayParameters(SOD, SDD);

    best_angle = [];
    for a in range(number_of_angles):
        best_angle.append(aPrediction[a+number_of_distances])

    pred_image = boneRotation(best_angle, 'All');

    return pred_image

def saveImageAndCSV(aPath, aTarget, aPredition, aMetricValue, aMetricName, aComputedTime, aScale):
    """
    Save single image and results.
    @Parameters:
        aPath: path to store the results
        aTarget: a target image
        aPredition: a set of predicted parameters (single image)
        aMetricValue: value of the metric (float, single value)
        aMetricName: name of the metric
        aComputedTime: total time cost to compute results
        aScale: Scaling factors
    """
    df = pd.DataFrame();
    pred_image = computePredictedImage(aPredition);
    target_image = aTarget;

    plt.imsave(aPath +"/pred.png", pred_image, cmap='Greys_r');
    plt.imsave(aPath+"/target.png", target_image, cmap='Greys_r');

    m = aMetricValue;
    row = [[aPredition, m[0], aComputedTime]];
    df2 = pd.DataFrame(row, columns=['Parameters', aMetricName, 'Time(s)']);
    df = df.append(df2, ignore_index=True);

    error_map = abs(target_image-pred_image);
    plt.imsave(aPath+"/error-map.png", error_map, cmap='Greys_r');

    correlation_map = target_image*pred_image;
    plt.imsave(aPath+"/correlation-map.png", correlation_map, cmap='Greys_r');

    df.to_csv(aPath+"/results.csv" );

    print("Image and csv file are saved");

def saveMultipleImageAndCSV(aPath, aTarget, aPredition, aMetricValue, aComputedTime, aScale):
    """
    Save multiple images and results.
    @Parameters:
        aPath: path to store the results
        aTarget: a target image
        aPredition: a set of predicted parameters (multiple images)
        aMetricValue: value of the metrics (1D array)
        aComputedTime: total time cost to compute results
        aScale: Scaling factors
    """
    df = pd.DataFrame();

    for r in range(len(aPredition[:,0])):

        target_image = aTarget;
        pred_image = computePredictedImage(aPredition[r,:]);
        
        plt.imsave(aPath +"/pred-%d.png" % r, pred_image, cmap='Greys_r');
        m = aMetricValue;
        row = [[r, aPredition[r,:], m[r,0], m[r,1], m[r,2], m[r,3], m[r,4], \
                m[r,5], m[r,6], m[r,7], aComputedTime]]
        df2 = pd.DataFrame(row, columns=['Image', 'Parameters', 'ZNCC', 'SSIM', 'MI', 'GC', \
                'MAE', 'CS', 'SSD', 'GD', 'Time(s)']);
        df = df.append(df2, ignore_index=True);

        error_map = abs(target_image-pred_image);
        plt.imsave(aPath +"/error-map-%d.png" % r, error_map, cmap='Greys_r');

        correlation_map = target_image*pred_image;
        plt.imsave(aPath +"/correlation-map-%d.png" % r, correlation_map, cmap='Greys_r');
    df.to_csv(aPath +"/results.csv" );

    print("Image and csv file are saved");

def computeAllMetricsValue(aTarget, aPredition):

    pred_image = computePredictedImage(aPredition);
    target_image = aTarget;

    ZNCC = -zncc(target_image, pred_image);
    SSIM = -ssim(target_image, pred_image);
    MI = -mi(target_image, pred_image);
    GC = -gc(target_image, pred_image);
    MAE = mae(target_image, pred_image);
    CS = cs(target_image, pred_image);
    SSD = ssd(target_image, pred_image);
    GD = gd(target_image, pred_image);

    return [ZNCC, SSIM, MI, GC, MAE, CS, SSD, GD]

def strArrayToFloatArray(aString):

    s = aString.replace('PARAMS', '');
    s = s.replace('[', '');
    s = s.replace(']', '');
    to_float = np.fromstring(s, dtype=float, sep=",")

    return to_float
