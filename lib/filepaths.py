#!/usr/bin/env python3
# -*- coding: utf-8 -*-



########################################################
#		File paths
########################################################



# built in
from itertools import combinations
import pickle

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



def build_file_paths() : 
	""" build and return all neeeded filepaths"""

	# folders 
	DATA_FOLDER = "./data"
	TEMP_FOLDER = "./temp"
	RESULT_FOLDER = "./results"

	# filenames 
	DATA_FILENAME = "bitstamp_last_1_5_year_1H.csv"
	CLEANED_DF_FILENAME = "cleaned_df_temp.csv"
	PREPARED_DF_FILENAME = "prepared_df_temp.csv"
	RESULT_BIN_FILENAME= "results.pk"
	RESULT_DF_FILENAME = "results.csv"

	# files
	DATA_FILE = DATA_FOLDER + DATA_FILENAME
	CLEANED_DF_FILE = TEMP_FOLDER + CLEANED_DF_FILENAME
	PREPARED_DF_FILE =RESULT_FOLDER + PREPARED_DF_FILENAME
	RESULT_BIN_FILE = RESULT_FOLDER + RESULT_BIN_FILENAME
	RESULT_DF_FILE = RESULT_FOLDER + RESULT_DF_FILENAME

	return DATA_FILE, CLEANED_DF_FILE, PREPARED_DF_FILE, RESULT_BIN_FILE, RESULT_DF_FILE
