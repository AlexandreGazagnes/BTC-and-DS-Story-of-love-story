
######################################################
######################################################
#	
######################################################
######################################################



########################################################
#		Description
########################################################


# This script is a kernel about doing machine learning with bitcoin prices



########################################################
#		Imports and configs
########################################################


# built in
from itertools import combinations
import pickle, datetime

# data management
import pandas as pd
import numpy as np
from stockstats import StockDataFrame

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# graphical config
sns.set()
# %matplotlib

# lib
from lib.filepaths import *
from lib.preprocessing import *
from lib.processing import *
from lib.results import *
from lib.time import * 



########################################################
#		Main
########################################################


# File paths
####################################

DATA_FILE, CLEANED_DF_FILE, \
	PREPARED_DF_FILE, RESULT_BIN_FILE, RESULT_DF_FILE = build_file_paths()

DATA_FILE = "/media/alex/CL/bitstamp_last_5_year_1H.csv"



# Constants and Meta params
#####################################

# meta results format
META_RESULTS_COL = ["score", "target", "period", "results", "grid"]

# features prices
P, NUMB = 12, 6
FEAT_PRICES = [(i+1) * P for i in range(NUMB)]

# features MAs
MA_TYPE, MA_PERIOD = "sma", [100., 300., 600., 1000.]

# target prices
TARGET_PRICES = [2*P,P]

# ML params
MODEL_LIST = [RandomForestClassifier]
MODEL_PARAMS = [	{'n_estimators': [300], 'bootstrap': [True], 
					'oob_score':  [True], 'warm_start':  [True]},]
TEST_SIZE = [0.25,]
CV = 5

# results params	
DROP_RESULTS_COLS = ["results", "grid"]

# timestamp
TIMESTAMP_X_TRAIN = [((8,10,2013),(5,12,2013))]
TIMESTAMP_X_TEST = [((8,10,2017),(18,12,2017))]
period = "ultra_euphoria"
bts = build_timestamp



# Results
#####################################

# initiate and save meta_results 
results = Results()


# Pre-processing
#####################################

# intiate and clean dataframe
df = DataFrame()
df.prepare(period="H").clean()

df.select(start=(1,12,2011), stop=(1,12, 2017))
df.plot()

df.featurize()
df.targetize()



# Processing
#####################################

# meta parametres
target = targets[1]
test_size = TEST_SIZE[0]
Model = MODEL_LIST[0]
model_params = MODEL_PARAMS[0]

# model init and fit
model = Model()
model.split_X_y().test_train_split()
model.fit()

# manage predictions and result
model.eval()

# update meta_results
results.meta.update(model.result, META_RESULTS_COL, RESULT_BIN_FILE)



# Analysing Results
######################################

# gather meta_results bin and csv
print(results.meta_)
results.meta_.graph()

# handle predict proba impact
results.meta.add_meta_probs_thresh(0.85)
results.build_meta_probs_param("period")

results.meta.probs.graph()
