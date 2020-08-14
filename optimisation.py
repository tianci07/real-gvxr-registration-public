#!/usr/bin/env python3

from pymoo.algorithms.so_cmaes import CMAES
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
import pandas as pd

import gvxrPython3 as gvxr
from utils import *
from metrics import *

import matplotlib.pyplot as plt
import cv2
import time
import argparse
import os

global metric_lists

metric_lists = {
    "ZNCC": zncc,
    "SSIM": ssim,
    "MI": mi,
    "GC": gc,
    "MAE": mae,
    "CS": cs,
    "SSD": ssd,
    "GD": gd
}

def getTargetImage(aTarget, aScale):
    """Get a ground truth image"""

    # read target image
    target_image = cv2.imread("./"+aTarget, 0);

    # zero-mean normalisation
    target_image = (target_image-target_image.mean())/target_image.std();

    #set nan and inf to zero
    target_image[np.isnan(target_image)]=0.;
    target_image[target_image > 1E308] = 0.;
    target_image=np.float32(target_image);

    if aScale != 0:
        # image pyramids - down scale
        for p in range(aScale):
            target_image = cv2.pyrDown(target_image);

    return target_image

def getSingleMetric(aPrediction):
    """Get single value of chosen metric"""
    obj_list = [];
    down_scale = int(args.downscale);

    for s in range(len(aPrediction[:, 0])):

        pred_image = computePredictedImage(aPrediction[s, :]);
        target_image = target;

        obj_value = metric_lists[single_metric](target_image, pred_image);
        obj_list.append(obj_value);

    return obj_list

def getAllMetrics(aPrediction):
    """Make use of all available metrics"""
    down_scale = int(args.downscale);
    obj_list = [];
    obj_list_temp = [];

    for s in range(len(aPrediction[:, 0])):

        pred_image = computePredictedImage(aPrediction[s, :]);
        target_image = target;

        if down_scale != 0:
            # image pyramids - down scale
            for p in range(down_scale):
                pred_image = cv2.pyrDown(pred_image);
                target_image = cv2.pyrDown(target);

        for key in metric_lists:
            obj_value = metric_lists[key](target_image, pred_image);
            obj_list_temp.append(obj_value);

        obj_list.append(obj_list_temp);

    return obj_list

class objectiveFunction(Problem):
    """Hand Registration with single objective function.
    Check performance of single metrics.
    """
    def __init__(self):
        super().__init__(n_var=38, n_obj=1, n_constr=0, type_var=np.float32);
        self.xl = np.array([0.7, 10., -90., -90., -90., -20., -90., -20., -20., -90., -90., -90.,
                            -20., -90., -90., -90., -20., -90., -90. ,-90., -20., -90., -90., -90.,
                            0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                            0.9, 0.9]);
        self.xu = np.array([0.95, 1000., 90., 90., 90., 20., 0., 20., 20., 0., 0., 0.,
                            20., 0., 0., 0., 20., 0., 0., 0., 20., 0., 20., 0.,
                            1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
                            1.1, 1.1]);

    def _evaluate(self, x, out, *args, **kwargs):

        objective = getSingleMetric(x);
        out["F"] = np.array(objective);
        print("FITNESS", np.min(out["F"]));


class multiObjectiveFunction(Problem):
    """Hand Registration with multiple objective functions.
    Check performance of optimisation using multiple metrics together.
    """
    def __init__(self):
        super().__init__(n_var=38, n_obj=8, n_constr=0, type_var=np.float32);
        self.xl = np.array([0.7, 10., -90., -90., -90., -20., -90., -20., -20., -90., -90., -90.,
                            -20., -90., -90., -90., -20., -90., -90. ,-90., -20., -90., -90., -90.,
                            0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                            0.9, 0.9]);
        self.xu = np.array([0.95, 1000., 90., 90., 90., 20., 0., 20., 20., 0., 0., 0.,
                            20., 0., 0., 0., 20., 0., 0., 0., 20., 0., 20., 0.,
                            1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
                            1.1, 1.1]);

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] = getAllMetrics(x);

def runCMAES(aPopulation, aGeneration, anInitialGuess):
    """CMA-ES algorithm"""

    pop_size = int(aPopulation);
    n_gen = int(aGeneration);
    initial_guess = strArrayToFloatArray(anInitialGuess);

    if int(args.downscale) != 0:
        pop_size = int(pop_size/2**int(args.downscale));
        n_gen = int(n_gen/2**int(args.downscale));

    # problems to be solved
    problem = objectiveFunction();

    if int(args.downscale) == 4:

        # use CMA-ES
        algorithm = CMAES(popsize=pop_size,
                        CMA_stds=0.1);
    else:
        # use CMA-ES
        algorithm = CMAES(x0=initial_guess,
                        popsize=pop_size,
                        CMA_stds=0.1);
    # record time
    start = time.time();

    # set up the optimiser
    res = minimize(problem,
                   algorithm,
                   ('n_iter', n_gen),
                   verbose=False);

    end = time.time();
    total_time = end-start;

    target_image = target;
    # save results
    saveImageAndCSV(full_path, target_image, res.X, res.F, single_metric, total_time, int(args.downscale));

    global nb_pop, nb_generations, runtime, metric_value, parameters;
    nb_pop.append(pop_size);
    nb_generations.append(n_gen);
    runtime.append(total_time);
    metric_value=computeAllMetricsValue(target, res.X);
    for x in range(len(res.X)):
        parameters.append(res.X[x]);

def runNSGA2(aPopulation, aGeneration):
    """NSGA-II algorithm"""
    pop_size = int(aPopulation);
    n_gen = int(aGeneration);
    # problems to be solved
    problem = multiObjectiveFunction();

    # use NSGA-II
    algorithm = NSGA2(
        pop_size=pop_size,
        eliminate_duplicates=True
    );

    # record time
    start = time.time();

    # set up the optimiser
    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   verbose=True);

    end = time.time();
    total_time = end-start;

    # save results
    saveMultipleImageAndCSV(full_path, target_image, res.X, res.F, total_time, int(args.downscale));

# add command-line argument option
parser = argparse.ArgumentParser(description='2D/3D registration by fast X-ray simulation and optimisation');

parser.add_argument("--target_image", help="Ground truth image", required=True);
parser.add_argument("--output_folder", help="Output folder to store results", required=True);
parser.add_argument("--algorithm", help="Choose algorithm, options= ['CMAES', 'NSGA-II']", required=True)
parser.add_argument("--repeat", help="Repeat running for how many times", required=True);
parser.add_argument("--metrics", help="Choose single or all metrics, options=['ZNCC', 'SSIM', 'MI', 'GC', 'MAE', 'CS', 'SSD', 'GD', 'ALL']", required=True);
parser.add_argument("--downscale", help="Lower the image resolution using Guassian Pyramids, eg. 0 for no scaling, 1 for 1/4 (1/2 width and 1/2 height), 2 for 1/16, so on...")
parser.add_argument("--pop_size", default=100, help="How many individuals are required (NSGA-II)?");
parser.add_argument("--gen_size", default=100, help="How many generations are required (NSGA-II)?");
parser.add_argument("--initial_guess", default=100, help="Initial guess", required=True);

args = parser.parse_args();

if args.metrics != "ALL":
    single_metric = args.metrics;

full_path = args.output_folder+"/"+args.algorithm+"/"+args.metrics+"/"+args.repeat+"/downscale-"+args.downscale;

if not os.path.exists(full_path):
    os.makedirs(full_path);

global target
# get ground truth image
target = getTargetImage(args.target_image, int(args.downscale));
file = open(args.initial_guess, "r");
file = file.read();
# set up simulation environment.
setXRayEnvironment(int(args.downscale));

np.random.seed();

pop_size = [];
nb_pop = [];
nb_generations = [];
runtime = [];
parameters = [];
fitness = [];
img_width = target.shape[1];
img_height = target.shape[0];

if args.algorithm == "CMAES":
    runCMAES(args.pop_size, args.gen_size, file);
elif args.algorithm == "NSGA2":
    runNSGA2(args.pop_size, args.gen_size);

print("METRICS", ',',
     args.metrics, ',',
     args.target_image, ',',
     args.algorithm, ',',
     int(args.downscale), ',',
     img_width, ',',
     img_height, ',',
     nb_pop[0], ',',
     nb_generations[0], ',',
     runtime[0], ',',
     metric_value[0], ',',
     metric_value[1], ',',
     metric_value[2], ',',
     metric_value[3], ',',
     metric_value[4], ',',
     metric_value[5], ',',
     metric_value[6], ',',
     metric_value[7]);

print("PARAMS", parameters);
