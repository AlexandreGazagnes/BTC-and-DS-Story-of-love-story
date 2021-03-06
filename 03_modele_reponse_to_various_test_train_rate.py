#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#####################################################################
#####################################################################
#	03_modele_reponse_to_various_test_train_rate
#####################################################################
#####################################################################



########################################################
#		Description
########################################################


# In this script, we will study the impact of the size of the train 
# and test dataset on the veracity of our predictions.
# We logically expect that the smaller the size of the dataset 
# stream, the worse our results are.



########################################################
#		Imports and configs
########################################################


# lib
from lib.filepaths import *
from lib.preprocessing import *
from lib.processing import *
from lib.results import *



########################################################
#		Main
########################################################


# File paths
####################################

DATA_FILE, CLEANED_DF_FILE, \
	PREPARED_DF_FILE, RESULT_BIN_FILE, RESULT_DF_FILE = build_file_paths()



# Constants and Meta params
#####################################

# meta results format
META_RESULTS_COL = ["score", "target", "test_size", "results", "grid"]

# features prices
P, NUMB = 12, 6
FEAT_PRICES = [(i+1) * P for i in range(NUMB)]

# features MAs
MA_TYPE, MA_PERIOD = "sma", [100., 300., 600., 1000.]

# target prices
TARGET_PRICES = [P, 2*P]

# ML params
MODEL_LIST = [RandomForestClassifier]
MODEL_PARAMS = [{'n_estimators': [300], 'bootstrap': [True], 'oob_score':  [True], 'warm_start':  [True]}]
TEST_SIZE  = np.arange(0.2, 0.9, 0.2)
CV = 5

# results params	
DROP_RESULTS_COLS = ["results", "grid"]



# Pre-processing
#####################################

# initiate and save meta_results 
meta_results = create_meta_results(META_RESULTS_COL)
save_bin_results(meta_results, RESULT_BIN_FILE)

# intiate and clean dataframe
df = create_df(DATA_FILE)
df = clean_data(df, CLEANED_DF_FILE)

# feat ingeenirng and targets
df = add_features(df, FEAT_PRICES, MA_TYPE, MA_PERIOD, PREPARED_DF_FILE)
df, targets = add_targets(df, TARGET_PRICES, PREPARED_DF_FILE)
df = drop_useless_feat(df, PREPARED_DF_FILE)



# Processing
#####################################

target = targets[1]
Model = MODEL_LIST[0]
model_params = MODEL_PARAMS[0]

for test_size in TEST_SIZE :

	# prepare X, y test and train
	X,y = create_X_y(df, target, targets)
	X_train, X_test, y_train, y_test \
		= split_X_y(X,y,test_size=test_size, shuffle=True, stratify=y) 

	# init and fit model
	grid = grid_init(Model, model_params, CV=CV)
	grid = grid_fit(grid, X_train, y_train)

	# manage results
	results = grid_results(grid, X_test,y_test)
	score = grid_score(results)

	# print brut results
	print("\nparams : {}".format(grid.best_params_))
	print("score avec {} sur la target {}: {:.4f}\n"
	  .format(Model.__name__, target, score))

	# # show base line
	# print("mais % de hausse sur l'echantillon : {:.4f}\n\n"
	# 	  .format(len(y[y == True])/len(y)))

	# update meta_results
	new_result = [score, target, test_size, results, grid.best_params_]
	update_results(new_result, META_RESULTS_COL, RESULT_BIN_FILE)



# Analysing Results
######################################

# gather meta_results bin and csv
meta_results = load_bin_results(RESULT_BIN_FILE)	
meta_results_ = readablize_results(meta_results, DROP_RESULTS_COLS)
meta_results_.to_csv(RESULT_DF_FILE, index= False)

# handle predict proba impact
create_meta_probs(meta_results, threshold=0.85)
meta_results_ = readablize_results(meta_results, DROP_RESULTS_COLS)
meta_probs, meta_quants = create_meta_probs(meta_results, param="test_size", graph=True)
