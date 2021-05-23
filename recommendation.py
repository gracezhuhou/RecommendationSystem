"""
Method Description:
Used a weighted hybrid RS combining Item-based CF RS with Pearson similarity and Model-based RS using XGBregressor.
Things I did to improve the accuracy:
1. Selected more features from provided datasets to train XGBregressor model.
2. Tuned XGBRegressor Parameters, select optimum max_depth.
3. Selected a better weight for hybrid RS

Error Distribution:
>=0 and <1: 102098
>=1 and <2: 33020
>=2 and <3: 6168
>=3 and <4: 758
>=4: 0

RSME:
0.9784973315841181

Execution Time: 259
"""

import sys
import math
import itertools
import os
from pyspark import SparkContext
import json
from operator import add
import datetime
import pandas as pd
import xgboost as xgb


start_time = datetime.datetime.now()

sc = SparkContext('local[1]', 'competition')
sc.setLogLevel('ERROR')
# folder = sys.argv[1]
# test_fp = sys.argv[2]
# output_fp = sys.argv[3]
folder = "publicdata"
test_fp = "publicdata/yelp_val_in.csv"
output_fp = "output.csv"

train_fp = os.path.join(folder, "yelp_train.csv")
fp_business = 'business.json'
fp_check = 'checkin.json'
fp_photo = 'photo.json'
fp_review = 'review_train.json'
fp_tip = 'tip.json'
fp_user = 'user.json'

data = sc.textFile(train_fp).zipWithIndex().filter(lambda x: x[1] != 0).keys()\
    .map(lambda x: x.split(","))

busi_user = data.map(lambda x: (x[1], x[0]))  # (b, u)
all_user_map = busi_user.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
busi_sets = busi_user.map(lambda x: (x[0], all_user_map[x[1]])).groupByKey().mapValues(set)

busi_avg_map = data.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(list)\
    .mapValues(lambda x: sum(x)/len(x)).collectAsMap()
user_avg_map = data.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(list)\
    .mapValues(lambda x: sum(x)/len(x)).collectAsMap()

busi_sets_map = busi_sets.collectAsMap()
rate_map = data.map(lambda x: ((all_user_map[x[0]], x[1]), float(x[2]))).collectAsMap()

user_num = len(all_user_map)
h_num = 100  # number of hash function


def h(i, row):
    a = i + 100
    b = i * 3
    return (a * row + b) % user_num


def min_hash(x):
    m = [user_num] * h_num
    for row in x[1]:
        for i in range(h_num):
            if h(i+1, row) < m[i]:
                m[i] = h(i, row)
    return x[0], m


sigs = busi_sets.map(min_hash)  # signature matrix


# LSH
b = 50  # bands
r = h_num // b  # rows
s_thre = 0.8  # similarity threshold


def sig2band(x):
    sig = x[1]
    bands = []
    for i in range(b):
        bands.append((i, sig[i*r: i*r+r]))
    return x[0], bands


def get_prediction(iter):
    size = 20000
    # buckets = [[] for _ in range(size)]
    buckets = {}
    i = 0
    for band in iter:
        i += 1
        h_value = hash(str(band[1]))
        index = h_value % size
        # buckets[index].append(band[0])
        if index in buckets:
            buckets[index].add(band[0])
        else:
            buckets[index] = {band[0]}
    # print(len(buckets))
    pred_list = []
    for bucket in buckets.values():
        w_dict = {}
        for pair in itertools.combinations(bucket, 2):
            b1, b2 = pair
            # weight
            corated_user = busi_sets_map[pair[0]].intersection(busi_sets_map[pair[1]])
            cnt = len(corated_user)
            if cnt < 10:
                continue
            # print(cnt)
            r1_list = []
            r2_list = []
            sum1 = sum2 = 0
            for u in corated_user:
                r1 = rate_map[(u, b1)]
                r2 = rate_map[(u, b2)]
                r1_list.append(r1)
                r2_list.append(r2)
                sum1 += r1
                sum2 += r2
            avg1 = sum1 / cnt
            avg2 = sum2 / cnt

            product = n1 = n2 = 0
            for i in range(cnt):
                r1_ = r1_list[i] - avg1
                r2_ = r2_list[i] - avg2
                product += r1_ * r2_
                n1 += r1_ * r1_
                n2 += r2_ * r2_

            if n1 == 0 or n2 == 0:
                continue
            w = product / (math.sqrt(n1) * math.sqrt(n2))
            # if w <= 0:
            #     continue
            w = w * (abs(w)**1.5)  # case amplication
            w_dict.setdefault(b1, [])
            w_dict.setdefault(b2, [])
            w_dict[b1].append((b2, w))
            w_dict[b2].append((b1, w))

        neibor = 4
        # pred
        for b0 in w_dict:
            w_dict[b0] = sorted(w_dict[b0], key=lambda x: x[1], reverse=True)
            if b0 in test_busi_map:
                users = test_busi_map[b0]  # string
                for u in users:
                    rw_sum = w_sum = 0
                    for i, (bn, w) in enumerate(w_dict[b0]):
                        if i > neibor:
                            break
                        rate_key = (all_user_map[u], bn)
                        if rate_key not in rate_map:
                            continue
                        rate = rate_map[rate_key]
                        rw_sum += w * rate
                        w_sum += abs(w)
                    pred_r = rw_sum/w_sum if w_sum != 0 else 0
                    pred_list.append(((u, b0), pred_r))
    return pred_list


test_data = sc.textFile(test_fp).zipWithIndex().filter(lambda x: x[1] != 0).keys()\
    .map(lambda x: (x.split(',')[1], x.split(',')[0]))  # b, u
test_busi_map = test_data.groupByKey().mapValues(list).collectAsMap()


all_bands = sigs.map(sig2band).flatMapValues(lambda x: x).map(lambda x: (x[1][0], (x[0], x[1][1])))\
    .partitionBy(b, lambda x: x).map(lambda x: x[1])


def get_pred(x):
    key = (x[1], x[0])
    if key in prediction_map:
        rate = prediction_map[key]
        if rate > 0.0:
            return rate
    if x[0] in busi_avg_map:
        avg_r = busi_avg_map[x[0]]
    else:
        avg_r = user_avg_map[x[1]]
    return avg_r


prediction_map = all_bands.mapPartitions(get_prediction).groupByKey().mapValues(list)\
    .mapValues(lambda x: max(x)).collectAsMap()

all_prediction = test_data.map(get_pred).collect()


# Model
def get_busi_info(x):
    n1 = n2 = n3 = n4 = n5 = n6 = n7 = n8 = n9 = n10 = n11 = 0
    if "attributes" in x and x["attributes"] is not None:
        if "RestaurantsPriceRange2" in x["attributes"]:
            n1 = int(x["attributes"]["RestaurantsPriceRange2"])
        if "RestaurantsTableService" in x["attributes"] and x["attributes"]["RestaurantsTableService"] == "True":
            n2 = 1
        if "GoodForKids" in x["attributes"] and x["attributes"]["GoodForKids"] == "True":
            n3 = 1
        if "NoiseLevel" in x["attributes"]:
            if x["attributes"]["NoiseLevel"] == "quiet":
                n4 = 1
            elif x["attributes"]["NoiseLevel"] == "loud":
                n4 = -1
        if "RestaurantsReservations" in x["attributes"] and x["attributes"]["RestaurantsReservations"] == "True":
            n5 = 1
        if "RestaurantsTakeOut" in x["attributes"] and x["attributes"]["RestaurantsTakeOut"] == "True":
            n6 = 1
        if "BusinessAcceptsCreditCards" in x["attributes"] and x["attributes"]["BusinessAcceptsCreditCards"] == "True":
            n7 = 1
        if "BusinessParking" in x["attributes"]:
            n8 = len(x["attributes"]["BusinessParking"].split("True")) - 1
        if "Caters" in x["attributes"] and x["attributes"]["Caters"] == "True":
            n9 = 1
        if "GoodForMeal" in x["attributes"]:
            n10 = len(x["attributes"]["GoodForMeal"].split("True")) - 1
        if "HasTV" in x["attributes"] and x["attributes"]["HasTV"] == "True":
            n11 = 1
    return x["business_id"], [x["stars"], x["review_count"], x["is_open"], x["latitude"], x["longitude"],
                              n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11]


business_info = sc.textFile(os.path.join(folder,  fp_business)).map(lambda x: json.loads(x)).map(get_busi_info)
business_map = business_info.collectAsMap()


def get_checkin_info(x):
    checkin_cnt = [0, 0, 0, 0, 0, 0, 0]
    for time in x["time"]:
        if time[:3] == "Mon":
            checkin_cnt[0] += x["time"][time]
        elif time[:3] == "Tue":
            checkin_cnt[1] += x["time"][time]
        elif time[:3] == "Wed":
            checkin_cnt[2] += x["time"][time]
        elif time[:3] == "Thu":
            checkin_cnt[3] += x["time"][time]
        elif time[:3] == "Fri":
            checkin_cnt[4] += x["time"][time]
        elif time[:3] == "Sat":
            checkin_cnt[5] += x["time"][time]
        elif time[:3] == "Sun":
            checkin_cnt[6] += x["time"][time]
    return x["business_id"], checkin_cnt


checkin_info = sc.textFile(os.path.join(folder,  fp_check)).map(lambda x: json.loads(x)).map(get_checkin_info)
checkin_map = checkin_info.collectAsMap()

photo_info = sc.textFile(os.path.join(folder,  fp_photo)).map(lambda x: json.loads(x))\
    .map(lambda x: (x["business_id"], 1)).reduceByKey(add)
photo_map = photo_info.collectAsMap()

tip_info = sc.textFile(os.path.join(folder,  fp_tip)).map(lambda x: json.loads(x))
tip_busi_info = tip_info.map(lambda x: (x["business_id"], 1)).reduceByKey(add)
tip_user_info = tip_info.map(lambda x: (x["user_id"], 1)).reduceByKey(add)
tip_busi_map = tip_busi_info.collectAsMap()
tip_user_map = tip_user_info.collectAsMap()


def get_user_info(x):
    n1 = int(datetime.datetime.strptime(x["yelping_since"], "%Y-%m-%d").toordinal())
    if x["friends"] == "None":
        n2 = 0
    else:
        n2 = len(x["friends"].split(","))

    return x["user_id"], [x["review_count"], x["fans"], x["useful"], x["average_stars"], n1, n2]


user_info = sc.textFile(os.path.join(folder,  fp_user)).map(lambda x: json.loads(x)).map(get_user_info)
user_map = user_info.collectAsMap()


def get_info(x):
    res = []
    res.extend(business_map[x[0]])
    res.extend(checkin_map[x[0]] if x[0] in checkin_map else [0, 0, 0, 0, 0, 0, 0])
    res.append(photo_map[x[0]] if x[0] in photo_map else 0)
    res.append(tip_busi_map[x[0]] if x[0] in tip_busi_map else 0)
    res.append(tip_user_map[x[0]] if x[0] in tip_user_map else 0)
    res.extend(user_map[x[1]])
    return res


train_data = data.map(lambda x: (x[1], x[0], float(x[2])))
train_y = train_data.map(lambda x: x[2]).collect()
train_X = train_data.map(get_info).collect()

test_X = test_data.map(get_info).collect()

X = pd.DataFrame(train_X)
y = pd.DataFrame(train_y)
X_test = pd.DataFrame(test_X)
test = pd.read_csv(test_fp)


model_xgb = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.1, n_jobs=-1)
model_xgb.fit(X, y)

xgb_pred = model_xgb.predict(X_test)

test["prediction1"] = all_prediction
test["prediction2"] = xgb_pred

# weighted
prediction = []
alpha = 0.05
for index, row in test.iterrows():
    prediction.append(alpha * row["prediction1"] + (1 - alpha) * row["prediction2"])
test["prediction"] = prediction
test.to_csv(output_fp, index=False, columns=["user_id", "business_id", "prediction"])

end_time = datetime.datetime.now()
print('Duration: ', int((end_time-start_time).total_seconds()))
