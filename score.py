from pyspark import SparkContext
import sys
import datetime
import math
import pandas as pd
from operator import add

start_time = datetime.datetime.now()
sc = SparkContext('local[1]', 'score')
sc.setLogLevel('ERROR')
predict_fp = "output.csv"
true_fp = "publicdata/yelp_val.csv"

predict = sc.textFile(predict_fp).zipWithIndex().filter(lambda x: x[1] != 0).keys()\
    .map(lambda x: ((x.split(',')[0], x.split(',')[1]), float(x.split(',')[2])))
true = sc.textFile(true_fp).zipWithIndex().filter(lambda x: x[1] != 0).keys()\
    .map(lambda x: ((x.split(',')[0], x.split(',')[1]), float(x.split(',')[2])))

score = predict.join(true).map(lambda x: ((x[1][0]-x[1][1]) ** 2, 1))\
    .reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]))
score = math.sqrt(score[0]/score[1])
print(score)


def cal_distribution(x):
    if x < 1:
        return "<1", 1
    elif x < 2:
        return "<2", 1
    elif x < 3:
        return "<3", 1
    elif x < 4:
        return "<4", 1
    else:
        return ">=4", 1


distribution = predict.join(true).map(lambda x: abs(x[1][0]-x[1][1])).map(cal_distribution).reduceByKey(add)
print(distribution.collect())

