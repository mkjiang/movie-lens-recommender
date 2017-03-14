# Databricks notebook source
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from math import sqrt

# COMMAND ----------

# Loading structed text file
movielens = sc.textFile("/FileStore/tables/nh5e5pr71488780385523/ratings.csv")

# COMMAND ----------

print(movielens.first(), movielens.count())

# COMMAND ----------

# Cleaning data
clean_data = movielens.map(lambda x: x.split(','))
header = clean_data.first()
clean_data = clean_data.filter(lambda x: x != header) 
clean_data.take(10)

# COMMAND ----------

# Just checking the scale of the problem
users = clean_data.map(lambda x: int(x[0]))
users.distinct().count()

# COMMAND ----------

# And for the movies
movies = clean_data.map(lambda x: int(x[1]))
movies.distinct().count()

# COMMAND ----------

# Setting up ratings for training
ratings = clean_data.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))
ratings.take(10)

# COMMAND ----------

# Do a 7/3 split on the original data
train, test = ratings.randomSplit([0.7, 0.3])
print(train.count(), test.count())

# COMMAND ----------

# Cache data to speed up training
train.cache()
test.cache()

# COMMAND ----------

# Train using ALS
# Let's first using 
# rank: latent factors
# numIterations: times to repeat process
rank = 5
numIterations = 10
model = ALS.train(train, rank, numIterations)

# COMMAND ----------

# For product 31, find 10 Users to sell to
model.recommendUsers(31, 10)

# COMMAND ----------

# For user 2, find 10 products to promote
model.recommendProducts(2,10)

# COMMAND ----------

# Predict single product for single user
model.predict(2, 31)

# COMMAND ----------

# Predict multi-users and multi products
train_X = train.map(lambda x:(x[0], x[1]))

# Returns Ratings(user, item, prediction)
pred = model.predictAll(train_X)
pred.first()

# COMMAND ----------

# Get performance estimate
# Organize the data to make (user,product) as the key

train_dict = train.map(lambda x:((x[0], x[1]), x[2]))
pred_dict = pred.map(lambda x:((x[0], x[1]), x[2]))

# Join so data becomes ((user, prduct), (true_rating, pred_rating))
train_pred = train_dict.join(pred_dict)

# r[1][0] is true rating, r[1][1] is pred rating
MSE = train_pred.map(lambda x: (x[1][0] - x[1][1])**2).mean()
RMSE = sqrt(MSE)

RMSE

# COMMAND ----------

# Now let's evaluate the performance on test set

test_X = test.map(lambda x:(x[0], x[1]))
pred = model.predictAll(test_X)

test_dict = test.map(lambda x:((x[0], x[1]), x[2]))
pred_dict = pred.map(lambda x:((x[0], x[1]), x[2]))
test_pred = test_dict.join(pred_dict)

test_MSE = test_pred.map(lambda x:(x[1][0] - x[1][1])**2).mean()
test_RMSE = sqrt(test_MSE)

print("Root mean squared error = " + str(test_RMSE))

# COMMAND ----------

# Save model for later use
model.save(sc, ".../out/movielens-recommender-model")
