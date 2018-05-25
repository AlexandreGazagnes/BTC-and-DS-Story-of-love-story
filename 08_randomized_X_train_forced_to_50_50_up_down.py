#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#####################################################################
#####################################################################
# 	08_randomized_X_train_forced_to_50_50_up_down
#####################################################################
#####################################################################



########################################################
#		Description
########################################################


# In this script, we will attempt to make real predictions by forcing 
# the model to learn on 50% of True targets and # 50% of False targets.
# This should minimize the impact of the selected dataset period and will be useful for 
# later.



########################################################
#		Import
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

DATA_FILE = "/media/alex/CL/bitstamp_last_5_year_1H.csv"



# Constants and Meta params
#####################################

# meta results format
META_RESULTS_COL = ["score", "target", "results", "grid"]

# features prices
P, NUMB = 12, 6
FEAT_PRICES = [(i+1) * P for i in range(NUMB)]

# features MAs
MA_TYPE, MA_PERIOD = "sma", [100., 300., 600., 1000.]

# target prices
TARGET_PRICES = [P, 2*P]

# ML params
MODEL_LIST = [RandomForestClassifier]
MODEL_PARAMS = [{'n_estimators': [300,], 'bootstrap': [True], 'oob_score':  [True], 'warm_start':  [True]}]
TEST_SIZE = [0.25]
CV = 5

# results params	
DROP_RESULTS_COLS = ["results", "grid"]



# Results
####################################

# initiate and save meta_results 
meta_results = create_meta_results(META_RESULTS_COL)
save_bin_results(meta_results, RESULT_BIN_FILE)



# Pre-processing X_train
#####################################

# intiate and clean dataframe
df = create_df(DATA_FILE)
df = clean_data(df, CLEANED_DF_FILE)

# plot
fig, (ax0, ax1) = plt.subplots(1,2)
ax0.plot(pd.to_datetime(df.timestamp, unit="s"), df.close)
ax0.set_xlabel("time"); ax0.set_ylabel("price in $")
ax0.set_title("Train Dataset")


# feat engineering and targets
df = add_features(df, FEAT_PRICES, MA_TYPE, MA_PERIOD, PREPARED_DF_FILE)
df, targets = add_targets(df, TARGET_PRICES, PREPARED_DF_FILE)
df = drop_useless_feat(df, PREPARED_DF_FILE)



# Pre-processing X_test
#####################################

# intiate and clean dataframe
target_file="/media/alex/CL/kraken_data_lastmonth_H.csv"
df2 = create_df(target_file)
df2 = clean_data(df2, CLEANED_DF_FILE)

# plot
ax1.plot(pd.to_datetime(df2.timestamp, unit="s"), df2.close)
ax1.set_xlabel("time"); ax1.set_ylabel("price in $")
ax1.set_title("Test Dataset")
plt.show()

# feat engineering and targets
df2 = add_features(df2, FEAT_PRICES, MA_TYPE, MA_PERIOD, PREPARED_DF_FILE)
df2, targets = add_targets(df2, TARGET_PRICES, PREPARED_DF_FILE)
df2 = drop_useless_feat(df2, PREPARED_DF_FILE)



# Processing
#####################################

# meta params
target = targets[1]
test_size = TEST_SIZE[0]
Model = MODEL_LIST[0]
model_params = MODEL_PARAMS[0]


# keep 50% up 50% down for X
df = egalize_up_down(df, target) 

# prepare X, y test and train
X_train, y_train = create_X_y(df, target, targets)
X_test, y_test = create_X_y(df2, target, targets)

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
new_result = [score, target, results, grid.best_params_]
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
meta_probs, meta_quants = create_meta_probs(meta_results, graph=True)
