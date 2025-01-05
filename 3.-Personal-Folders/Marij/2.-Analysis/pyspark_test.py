# Create a PySpark session
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
spark = SparkSession.builder.master("local[*]").getOrCreate()

df = pd.read_csv('../4. Outputs/master_cleaned_csv.csv')
df = df[
    ["date", "home_club_position", "away_club_position", "attendance", "home_club_strategy",
     "away_club_strategy", "stadium_seats", "country_name", "home_club_average_age",
     "away_club_average_age", "home_club_average_height", "away_club_average_height",
     "home_club_average_market_value", "away_club_average_market_value", "home_club_goals",
     "away_club_goals"]].head(100)
print(f"{df.info()}")
        
df = spark.createDataFrame(df)
# Select the features and label columns
feature_columns = ["home_club_position", "away_club_position", "attendance"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Define the regression model
regression = LinearRegression(featuresCol="features", labelCol="home_club_goals")

# Set up a pipeline
pipeline = Pipeline(stages=[assembler, regression])

# Split the data into training and testing sets
(trainingData, testData) = df.randomSplit([0.7, 0.3])

# Train the model
model = pipeline.fit(trainingData)

# Make predictions
predictions = model.transform(testData)

# Select example rows to display
predictions.select("prediction", "home_club_goals", "features").show(5)