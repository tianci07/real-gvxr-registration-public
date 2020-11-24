#!/usr/bin/env python3

from pymoo.algorithms.so_cmaes import CMAES
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.factory import get_problem
from pymoo.visualization.scatter import Scatter

from utils import *
from metrics import *

import cv2
import time
import argparse
import os
import matplotlib.patches as patches

global metric_lists

metric_lists = {
    "ZNCC": zncc,
    "SSIM": ssim,
    "MI": mi,
    "GC": gc,
    "MAE": mae,
    "MAEROI":maeroi,
    "CS": cs,
    "SSD": ssd,
    "GD": gd
}

def getSingleMetric(aPrediction):
    """Get single value of chosen metric"""
    obj_list = []

    for s in range(len(aPrediction[:, 0])):

        pred_image = computePredictedImage(aPrediction[s, :])

        target_image = target

        if args.add_masks:
            pred_image = create_mask(pred_image, 0)
            target_image = create_mask(target_image, 250)

        if single_metric == 'MAEROI':
            roi = [[35, 411], [150, 127], [339, 15], [539, 63], [670, 235]]
            roi_multiplier = [0.1, 0.1, 0.1, 0.1, 0.1]

            obj_value = metric_lists[single_metric](target_image, pred_image, roi, 100, 100, roi_multiplier)
        else:
            obj_value = metric_lists[single_metric](target_image, pred_image)
        obj_list.append(obj_value)

    return obj_list

def getAllMetrics(aPrediction):
    """Make use of all available metrics"""
    down_scale = img_scale
    obj_list = []
    obj_list_temp = []

    two_metrics = ["ZNCC", "MAE"]
    print(aPrediction.shape)

    for s in range(len(aPrediction[:, 0])):

        pred_image = computePredictedImage(aPrediction[s, :])
        target_image = target

        # for key in metric_lists:
        #     obj_value = metric_lists[key](target_image, pred_image)
        #     obj_list_temp.append(obj_value)
        obj_list_temp_1 = zncc(target_image, pred_image)
        obj_list_temp_2 = mae(target_image, pred_image)

        obj_list_temp = [obj_list_temp_1, obj_list_temp_2]
        obj_list.append(obj_list_temp)

    return obj_list

class objectiveFunction(Problem):
    """Hand Registration with single objective function.
    Check performance of single metrics.
    """
    def __init__(self):
        super().__init__(n_var=38, n_obj=1, n_constr=0, type_var=np.float32)
        self.xl = np.array([0.7, 10., -20., -20., -20.,
                            -10., -20., -10.,
                            -10., -20., -20., -20.,
                            -10., -20., -20., -20.,
                            -10., -20., -20. ,-20.,
                            -10., -20., -20., -20.,
                            0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        self.xu = np.array([0.95, 1000., 20., 20., 20.,
                            10., 0., 10.,
                            10., 0., 0., 0.,
                            10., 0., 0., 0.,
                            10., 0., 0., 0.,
                            10., 0., 0., 0.,
                            1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
        # self.xl = np.array([0.7, 10., -90., -90., -90., -20., -90., -20., -20., -90., -90., -90.,
        #                     -20., -90., -90., -90., -20., -90., -90. ,-90., -20., -90., -90., -90.,
        #                     0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
        #                     0.9, 0.9])
        # self.xu = np.array([0.95, 1000., 90., 90., 90., 20., 0., 20., 20., 0., 0., 0.,
        #                     20., 0., 0., 0., 20., 0., 0., 0., 20., 0., 20., 0.,
        #                     1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
        #                     1.1, 1.1])

        self.xl[1] /= 1000
        self.xu[1] /= 1000
        for i in range(22):
            self.xl[i+2] /= 100
            self.xu[i+2] /= 100

        self.counter=-1

    def _evaluate(self, x, out, *args, **kwargs):

        # print("x.shape", x.shape)
        for i in range(x.shape[0]):
            x[i,1]*=1000
            for j in range(22):
                x[i,j+2]*=100

        objective = getSingleMetric(x)
        out["F"] = np.array(objective)
        print("FITNESS", np.min(out["F"]))
        resX = x[np.argmin(out["F"]),:]

        # res_pred = computePredictedImage(resX)
        # for scale in range(img_scale):
        #     res_pred = cv2.pyrUp(res_pred)
        # resF = metric_lists[single_metric](target_full_reso, res_pred)
        # print("FULLFITNESS", resF)

        resX_list = []
        for rx in range(len(resX)):
            resX_list.append(resX[rx])
        print("PARAMS", resX_list)
        self.counter+=1
        global call_count
        call_count = self.counter

class multiObjectiveFunction(Problem):
    """Hand Registration with multiple objective functions.
    Check performance of optimisation using multiple metrics together.
    """
    def __init__(self):
        super().__init__(n_var=38, n_obj=2, n_constr=0, type_var=np.float32)
        self.xl = np.array([0.7, 10., -90., -90., -90., -20., -90., -20., -20., -90., -90., -90.,
                            -20., -90., -90., -90., -20., -90., -90. ,-90., -20., -90., -90., -90.,
                            0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                            0.9, 0.9])
        self.xu = np.array([0.95, 1000., 90., 90., 90., 20., 0., 20., 20., 0., 0., 0.,
                            20., 0., 0., 0., 20., 0., 0., 0., 20., 0., 20., 0.,
                            1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
                            1.1, 1.1])
        # super().__init__(n_var=24, n_obj=2, n_constr=0, type_var=np.float32)
        # self.xl = np.array([0.7, 10., -90., -90., -90., -20., -90., -20., -20., -90., -90., -90., -20., -90., -90., -90., -20., -90., -90. ,-90., -20., -90., -90., -90.])
        # self.xu = np.array([0.95, 1000., 90., 90., 90., 20., 0., 20., 20., 0., 0., 0., 20., 0., 0., 0., 20., 0., 0., 0., 20., 0., 20., 0.])
        self.counter=0

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] = getAllMetrics(x)

        # Number of generations
        self.counter+=1
        global call_count
        call_count = self.counter

def runCMAES(aPopulation, aGeneration, anInitialGuess):
# def runCMAES(anInitialGuess):
    """CMA-ES algorithm"""

    initial_guess = strArrayToFloatArray(anInitialGuess, 'PARAMS', 'comma')

    # problems to be solved,
    problem = objectiveFunction()

    if (aPopulation != "NONE" and aGeneration != "NONE"):

        pop_size = int(aPopulation)
        gen_size = int(aGeneration)

    elif aPopulation != "NONE":
        pop_size = int(aPopulation)

    else:
        # number of individuals = 4+int(3*np.log(N)), N is number of input parameters
        pop_size = 4+int(3*np.log(total_number_of_params))

        # use CMA-ES with no fixed number of individuals and generations
        algorithm = CMAES(popsize=pop_size,
                        sigma=0.05,
                        restarts=2,
                        maxfevals=np.inf,
                        tolfun=1e-6,
                        restart_from_best=True,
                        incpopsize=2,
                        bipop=True)

    if img_scale != 0:
        pop_size = int(pop_size/2**img_scale)
        gen_size = int(gen_size/2**img_scale)

    if img_scale == 4:

        # use CMA-ES
        algorithm = CMAES(popsize=pop_size,
                        CMA_stds=0.1)
    else:

        if initial_guess.size != 0:

            initial_guess[1]/=1000
            for i in range(22):
                initial_guess[i+2]/=100

            # initial guess
            algorithm = CMAES(x0=initial_guess,
                            popsize=pop_size,
                            CMA_stds=0.1)
        else:

            # no initial guess
            algorithm = CMAES(popsize=pop_size,
                            sigma=0.05,
                            restarts=2,
                            maxfevals=np.inf,
                            tolfun=1e-6,
                            restart_from_best=True,
                            incpopsize=2,
                            bipop=True)
    # record time
    start = time.time()

    if aGeneration != "NONE":

        # set up the optimiser
        res = minimize(problem,
                       algorithm,
                       ('n_iters', gen_size),
                       verbose=False)
    else:
        # ('n_evals', 80000),
        seed = 10+total_number_of_runs
        res = minimize(problem,
                       algorithm,
                       seed=seed,
                       verbose=False)

    end = time.time()
    total_time = end-start
    target_image = target

    resX = np.array(res.X)
    resX[1]*=1000
    for j in range(22):
        resX[j+2]*=100

    if args.add_masks:
        target_image = (target_image-target_image.mean())/target_image.std()

    # save results
    saveImageAndCSV(full_path, target_image, resX, res.F, single_metric, total_time, img_scale)

    # collect all variables
    global nb_individuals, nb_generations, runtime, metric_value, parameters
    if aPopulation != "NONE":
        nb_individuals=len(res.pop)
    else:
        nb_individuals=pop_size
    nb_generations=call_count

    runtime=total_time
    metric_value=computeAllMetricsValue(target, resX, target_full_reso, img_scale)
    for x in range(len(resX)):
        parameters.append(resX[x])

def runNSGA2(aPopulation, aGeneration, anInitialGuess):
    """NSGA-II algorithm"""

    pop_size = int(aPopulation)
    gen_size = int(aGeneration)

    initial_guess = strArrayToFloatArray(anInitialGuess, 'PARAMS', 'comma')

    # problems to be solved
    problem = multiObjectiveFunction()

    # use NSGA-II
    # eliminate_duplicates=True
    algorithm = NSGA2(x0=initial_guess,
                    pop_size=pop_size)

    # record time
    start = time.time()

    # set up the optimiser
    res = minimize(problem,
                   algorithm,
                   ('n_gen', gen_size),
                   verbose=True)

    end = time.time()
    total_time = end-start
    target_image = target

    # save results
    saveMultipleImageAndCSV(full_path, target_image, res.X, res.F, total_time, img_scale)

    # collect all variables
    global nb_individuals, nb_generations, runtime, metrics, parameters

    nb_individuals = len(res.pop)
    nb_generations = call_count

    runtime = total_time

    # metric_value=computeAllMetricsValue(target, res.X, target_full_reso, img_scale)
    row,col = res.X.shape
    for x in range(row):
        parameters_temp = []
        for y in range(col):
            parameters_temp.append(res.X[x,y])
        parameters.append(parameters_temp)

    row,col = res.F.shape
    for f in range(row):

        metrics.append(res.F[f,:])


# add command-line argument option
parser = argparse.ArgumentParser(description='2D/3D registration by fast X-ray simulation and optimisation')

parser.add_argument("--target_image", help="Ground truth image", required=True)
parser.add_argument("--output_folder", help="Output folder to store results", required=True)
parser.add_argument("--algorithm", help="Choose algorithm, options= ['CMAES', 'NSGA-II']", required=True)
parser.add_argument("--repeat", help="Repeat running for how many times", required=True)
parser.add_argument("--metrics", help="Choose single or all metrics, options=['ZNCC', 'SSIM', 'MI', 'GC', 'MAE', 'CS', 'SSD', 'GD', 'ALL']", required=True)
parser.add_argument("--downscale", help="Lower the image resolution using Guassian Pyramids, eg. 0 for no scaling, 1 for 1/4 (1/2 width and 1/2 height), 2 for 1/16, so on...")
parser.add_argument("--pop_size", default="NONE", help="How many individuals are required (NSGA-II)?")
parser.add_argument("--gen_size", default="NONE", help="How many generations are required (NSGA-II)?")
parser.add_argument("--initial_guess", help="Initial guess", required=True)
parser.add_argument("--add_masks", help="added masks for both target and prediction", action='store_true')
args = parser.parse_args()

if args.metrics != "ALL":
    single_metric = args.metrics

if args.downscale == '0':
    full_path = args.output_folder+"/"+args.algorithm+"/"+args.metrics+"/"+args.repeat
else:
    full_path = args.output_folder+"/"+args.algorithm+"/"+args.metrics+"/"+args.repeat+"/downscale-"+args.downscale

if not os.path.exists(full_path):
    os.makedirs(full_path)

global target, img_scale, total_number_of_params

img_scale=int(args.downscale)
total_number_of_params = 38
total_number_of_runs = int(args.repeat)
# get ground truth image
if args.add_masks:
    target = cv2.imread("./"+args.target_image, img_scale);
    target_full_reso = cv2.imread("./"+args.target_image, 0);
else:
    target = getTargetImage(args.target_image, img_scale)
    target_full_reso = getTargetImage(args.target_image, 0)
file = open(args.initial_guess, "r")
file = file.read()

# set up simulation environment.
setXRayEnvironment(img_scale)
#
# np.random.seed()

pop_size = []
nb_generations = []
runtime = []
parameters = []
metrics = []
fitness = []
img_width = target.shape[1]
img_height = target.shape[0]

if args.algorithm == "CMAES":
    runCMAES(args.pop_size, args.gen_size, file)

    print("METRICS", ',',
         args.metrics, ',',
         args.target_image, ',',
         args.algorithm, ',',
         img_scale, ',',
         img_width, ',',
         img_height, ',',
         nb_individuals, ',',
         nb_generations, ',',
         runtime, ',',
         metric_value[0], ',',
         metric_value[1], ',',
         metric_value[2], ',',
         metric_value[3], ',',
         metric_value[4], ',',
         metric_value[5], ',',
         metric_value[6], ',',
         metric_value[7], ',',
         metric_value[8], ',',
         metric_value[9], ',',
         metric_value[10], ',',
         metric_value[11], ',',
         metric_value[12], ',',
         metric_value[13], ',',
         metric_value[14], ',',
         metric_value[15])

    # print("INITIALGUESS", parameters)

    print("GENERATIONS", nb_generations)

elif args.algorithm == "NSGA2":
    runNSGA2(args.pop_size, args.gen_size, file)

    for i in range(len(parameters)):

        print("MULTIPARAMS", parameters[i])

    for j in range(len(metrics)):

        print("FITNESS", metrics[j])
