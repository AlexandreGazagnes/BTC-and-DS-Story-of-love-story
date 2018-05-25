#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#####################################################################
#####################################################################
#	04_modele_reponse_to_extreme_unknown_scenari
#####################################################################
#####################################################################



########################################################
#		Description
########################################################



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

DATA_FILE = "/media/alex/CL/bitstamp_last_5_year_1H.csv"



# Constants and Meta params
#####################################

# meta results format
META_RESULTS_COL = ["score", "target", "period","results", "grid"]

# features prices
P, NUMB = 12, 6
FEAT_PRICES = [(i+1) * P for i in range(NUMB)]

# features MAs
MA_TYPE, MA_PERIOD = "sma", [100., 300., 600., 1000.]

# target prices
TARGET_PRICES = [P, 2*P]

# ML params
MODEL_LIST = [RandomForestClassifier]
MODEL_PARAMS = [{'n_estimators': [300, ], 'bootstrap': [True], 'oob_score':  [True], 'warm_start':  [True]}]
TEST_SIZE = [0.25]
CV = 5

# results params	
DROP_RESULTS_COLS = ["results", "grid"]

# timestamp
VARIOUS_TIMESTAMP_1 = [
 						('global_buble1', 1367369520, 1399528320),
						('euphoria1', 1367369520, 1386330720),
						('crisis1', 1386291120, 1420044720),
						('down_up1', 1389952320, 1485351840),
						('global_buble2', 1367369520, 1432313040),
						('down_up2', 1417416720, 1451619840),
						('global_buble3', 1435747440, 1522017840),
						('euphoria2', 1435747440, 1513611840),
						('ultra_euphoria', 1483238640, 1513611840),
						('global_buble4', 1483238640, 1522017840),
						('crisis2', 1513611840, 1518965040),
						('crisis3', 1513611840, 1522017840)]



# Pre-processing 
#####################################

# initiate and save meta_results 
meta_results = create_meta_results(META_RESULTS_COL)
save_bin_results(meta_results, RESULT_BIN_FILE)

# intiate and clean dataframe
df = create_df(DATA_FILE)
df = clean_data(df, CLEANED_DF_FILE)

plt.plot(pd.to_datetime(df.timestamp, unit="s"), df.close)
plt.xlabel("time"); plt.ylabel("price in $")
plt.title("Train Dataset")
plt.show()

# feat engineering and targets
df = add_features(df, FEAT_PRICES, MA_TYPE, MA_PERIOD, PREPARED_DF_FILE)
df, targets = add_targets(df, TARGET_PRICES, PREPARED_DF_FILE)
df = drop_useless_feat(df, PREPARED_DF_FILE)



# Processing
#####################################

target = targets[1]
test_size = TEST_SIZE[0]
Model = MODEL_LIST[0]
model_params = MODEL_PARAMS[0]

for period, start, stop in VARIOUS_TIMESTAMP_1 : 
	
	# prepare X, y test and train
	df_test = df.loc[start : stop, :]
	df_train = df.drop(df_test.index, axis=0, inplace=False, errors="ignore")
	X_train, y_train = create_X_y(df_train, target, targets)
	X_test, y_test = create_X_y(df_test, target, targets)

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
	new_result = [score, target, period, results, grid.best_params_]
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
meta_probs, meta_quants = create_meta_probs(meta_results, param="period", graph=True)


