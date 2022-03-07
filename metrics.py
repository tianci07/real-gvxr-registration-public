import numpy as np
import math
from skimage import metrics
from scipy.stats import chisquare, entropy
from skimage import filters
import random

def removeNullData(anArray):
    anArray[np.isnan(anArray)]=0.
    anArray[np.isinf(anArray)] = 0.

    return anArray

# Similarity metrics, use negative values
def nzncc(y_true, y_pred):
    """
    Zero-mean Normalised Cross Correlation.
    ZNCC = (1/n)*(1/(std(target_image)*std(est_image)))* SUM_n_by_n{(target_image-
            mean(target_image))*(est_image-mean(est_image))}
    @Parameters:
        y_true: ground truth or target image
        y_pred: predicted image
    """
    z = np.sum((y_true-y_true.mean())*(y_pred-y_pred.mean()))

    z /= -(y_true.shape[0]*y_true.shape[1]*y_true.std()*y_pred.std())
    if np.isnan(z)==True or np.isinf(z)==True:
        z = 0.

    return z

def nssim(y_true, y_pred):
    """
    Structural Similarity
    @Parameters:
        y_true: ground truth or target image
        y_pred: predicted image
    """
    s = -metrics.structural_similarity(y_true, y_pred, data_range=y_pred.max()-y_pred.min())

    if np.isnan(s)==True or np.isinf(s)==True:
        s = 0.

    return s

def nmi(y_true, y_pred):
    """
    Mutual information
    @Parameters:
        y_true: ground truth or target image
        y_pred: predicted image
    """

    return -metrics.normalized_mutual_information(y_true, y_pred)

def ngc(y_true, y_pred):
    """
    Gradient Correlation
    @Parameters:
        y_true: ground truth or target image
        y_pred: predicted image
    """

    # horizontal sobel filter
    ZNCC_h = zncc(filters.sobel(y_true,axis=0), filters.sobel(y_pred,axis=0))
    # vertical sobel filter
    ZNCC_v = zncc(filters.sobel(y_true,axis=1), filters.sobel(y_pred,axis=1))

    return -(ZNCC_h+ZNCC_v)/2

def nsrc(y_true, y_pred):
    """
    Stochastic Rank Correlation
    @Parameters:
        y_true: ground truth or target image
        y_pred: predicted image
    """
    r = 20 # example selected region [20, 20]
    rn = random.randint(r, min(y_true.shape[0],y_true.shape[1])-20)

    y_true = y_true[rn:rn+r,rn:rn+r]
    y_pred = y_pred[rn:rn+r,rn:rn+r]

    y_true_rank = y_true.argsort()
    y_pred_rank = y_pred.argsort()

    return (6*np.sum(y_true_rank-y_pred_rank)**2/(r*(r**2-1)))

# Dis-similarity metrics
def mae(y_true, y_pred):
    """
    Mean Absolute Error
    @Parameters:
        y_true: ground truth or target image, type=float array
        y_pred: predicted image, type=float array
    """
    y_true = removeNullData(y_true)
    y_pred = removeNullData(y_pred)
    
    return np.mean(np.abs(y_true - y_pred))

def cs(y_true, y_pred):
    """
    Chi-square distance
    @Parameters:
        y_true: ground truth or target image, type=float array
        y_pred: predicted image, type=float array
    """
    chiq, p_value = chisquare(np.ravel(y_pred), f_exp=np.ravel(y_true))

    return chiq

def ssd(y_true, y_pred):
    """
    Sum of Square Difference
    @Parameters:
        y_true: ground truth or target image, type=float array
        y_pred: predicted image, type=float array
    """
    return np.sum(np.square(y_true - y_pred))

def rmse(y_true, y_pred):
    
    return math.sqrt(np.mean(np.square(y_true - y_pred)))

def gd(y_true, y_pred):
    """
    Gradient Difference
    @Parameters:
        y_true: ground truth or target image, type=float array
        y_pred: predicted image, type=float array
    """
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # horizontal sobel filter
    sobel_diff_h = np.sum(filters.sobel(y_true,axis=0)-filters.sobel(y_pred,axis=0))
    # vertical sobel filter
    sobel_diff_v = np.sum(filters.sobel(y_true,axis=1)-filters.sobel(y_pred,axis=1))

    return (var_true/(var_true+sobel_diff_h**2) + var_pred/(var_pred+sobel_diff_v**2))

def pi(y_true, y_pred):
    """
    Pattern Intensity
    @Parameters:
        y_true: ground truth or target image, type=float array
        y_pred: predicted image, type=float array
    """

    s = 10# sigma
    r = 3 # radius
    sum_list=[]

    for i in range(r, y_true.shape[0]-r-1):
        for j in range(r-1, y_true.shape[1]-r-1):
            sum = np.sum(y_true[i-3:i+4, j-3:j+4]-y_pred[i-3:i+4, j-3:j+4])
            sum_list.append(sum)

    return (s**2/(s**2 + np.sum(sum_list)**2))
