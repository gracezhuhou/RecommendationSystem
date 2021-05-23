# RecommendationSystem

Method Description:  
Used a weighted hybrid RS combining Item-based CF RS with Pearson similarity and Model-based RS using XGBregressor.  
Data comes from public data of Yelp.  
Things I did to improve the accuracy:  
1. Selected more features from provided datasets to train XGBregressor model.  
2. Tuned XGBRegressor Parameters, select optimum max_depth.  
3. Selected a better weight for hybrid RS.  
Error Distribution:  
>=0 and <1: 102098  
>=1 and <2: 33020  
>=2 and <3: 6168  
>=3 and <4: 758  
>=4: 0  
RSME:  
0.9784973315841181  
Execution Time: 259  
