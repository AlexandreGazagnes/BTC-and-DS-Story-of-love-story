#!/usr/bin/env python3
# -*- coding: utf-8 -*-



########################################################
#		Time
########################################################



import datetime



def build_timestamp(d=0, m=1,y=2000) : 
	return int(datetime.datetime(y,m,d).timestamp())
