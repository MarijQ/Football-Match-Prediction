# Plan
# 1. Imports
# 2. File imports and merging
# 3. Feature engineering
# 4. Data cleaning
# 5. EDA
# 6. Machine learning

# 1. Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, year
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier, DecisionTreeClassifier, \
    RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# os.environ['HADOOP_HOME'] = 'C:\\hadoop'
# os.environ['SPARK_HOME'] = r"C:\opt\spark\spark-3.5.1-bin-hadoop3.tgz"
# os.environ['PYSPARK_SUBMIT_ARGS'] = '--master local[*] pyspark-shell'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)


class DDA():
    def __init__(self):
        self.spark = SparkSession.builder.appName("DataDrivenApproach").getOrCreate()

    def ML_load(self):
        self.df = pd.read_csv('../4. Outputs/master_cleaned_csv.csv')
        self.df = self.df[
            ["date", "home_club_position", "away_club_position", "attendance", "home_club_strategy",
             "away_club_strategy", "stadium_seats", "country_name", "home_club_average_age",
             "away_club_average_age", "home_club_average_height", "away_club_average_height",
             "home_club_average_market_value", "away_club_average_market_value", "home_club_goals",
             "away_club_goals"]]
        print(f"{self.df.info()}")
        # print(f"{self.df.shape}")
        # print("\n", self.df.isnull().sum(), "\n")
        # rest of code goes here
        pass

    def preprocess_data(self):
        # Convert the pandas DataFrame to a PySpark DataFrame
        self.df = self.spark.createDataFrame(self.df)

        # Convert 'date' to integer year
        self.df = self.df.withColumn('year', year(col('date')))
        self.df = self.df.drop('date')

        # Create 'outcome' column based on 'home_club_goals' and 'away_club_goals'
        self.df = self.df.withColumn('outcome',
                                     when(col('home_club_goals') > col('away_club_goals'), 'home')
                                     .when(col('home_club_goals') < col('away_club_goals'), 'away')
                                     .otherwise('draw'))

        # Drop 'home_club_goals' and 'away_club_goals'
        self.df = self.df.drop('home_club_goals', 'away_club_goals')

        # Identify categorical columns for one-hot encoding
        categorical_cols = ['home_club_strategy', 'away_club_strategy', 'country_name']
        stages = []

        # StringIndexer and OneHotEncoder for each categorical column
        for col_name in categorical_cols:
            indexer = StringIndexer(inputCol=col_name, outputCol=col_name + '_Index')
            encoder = OneHotEncoder(inputCols=[indexer.getOutputCol()],
                                    outputCols=[col_name + '_Vec'])
            stages += [indexer, encoder]

        # Outcome label indexer
        label_indexer = StringIndexer(inputCol='outcome', outputCol='label').fit(self.df)
        stages += [label_indexer]

        # Assemble vectors
        numerical_cols = ['year', 'home_club_position', 'away_club_position', 'attendance',
                          'stadium_seats',
                          'home_club_average_age', 'away_club_average_age',
                          'home_club_average_height',
                          'away_club_average_height', 'home_club_average_market_value',
                          'away_club_average_market_value']
        assembler_inputs = [c + '_Vec' for c in categorical_cols] + numerical_cols
        assembler = VectorAssembler(inputCols=assembler_inputs, outputCol='features')
        stages += [assembler]

        # Create a pipeline
        pipeline = Pipeline(stages=stages)
        pipeline_model = pipeline.fit(self.df)
        self.df = pipeline_model.transform(self.df)

        # Select features and label for model training
        self.df = self.df.select('features', 'label')

    def train_test_split(self):
        # Split the data into training and test sets (80:20)
        self.train_df, self.test_df = self.df.randomSplit([0.8, 0.2], seed=42)

    def train_models(self):
        accuracy_evaluator = MulticlassClassificationEvaluator(labelCol='label',
                                                               predictionCol='prediction',
                                                               metricName='accuracy')
        precision_evaluator = MulticlassClassificationEvaluator(labelCol='label',
                                                                predictionCol='prediction',
                                                                metricName='weightedPrecision')
        recall_evaluator = MulticlassClassificationEvaluator(labelCol='label',
                                                             predictionCol='prediction',
                                                             metricName='weightedRecall')

        # Neural Network
        layers = [self.train_df.schema['features'].metadata['ml_attr']['num_attrs']] + [5, 5, 3]
        nn = MultilayerPerceptronClassifier(layers=layers, seed=42)
        nn_model = nn.fit(self.train_df)
        nn_predictions = nn_model.transform(self.test_df)

        # Decision Tree
        dt = DecisionTreeClassifier(labelCol='label', featuresCol='features', seed=42)
        dt_model = dt.fit(self.train_df)
        dt_predictions = dt_model.transform(self.test_df)

        # Random Forest
        rf = RandomForestClassifier(labelCol='label', featuresCol='features', seed=42)
        rf_model = rf.fit(self.train_df)
        rf_predictions = rf_model.transform(self.test_df)

        # Logistic Regression
        lr = LogisticRegression(labelCol='label', featuresCol='features', maxIter=10)
        lr_model = lr.fit(self.train_df)
        lr_predictions = lr_model.transform(self.test_df)

        # Evaluate models
        models = {
            'Neural Network': (nn_predictions, nn_model),
            'Decision Tree': (dt_predictions, dt_model),
            'Random Forest': (rf_predictions, rf_model),
            'Logistic Regression': (lr_predictions, lr_model)
        }

        for model_name, (predictions, model) in models.items():
            accuracy = accuracy_evaluator.evaluate(predictions)
            precision = precision_evaluator.evaluate(predictions)
            recall = recall_evaluator.evaluate(predictions)

            print(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

            # Cross-validation can be used instead of simple train-test split if preferred
            # Uncomment the following lines to use CrossValidator
            # paramGrid = ParamGridBuilder().build()
            # crossval = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=accuracy_evaluator, numFolds=3)
            # cvModel = crossval.fit(self.train_df)
            # cv_predictions = cvModel.transform(self.test_df)
            # acc = accuracy_evaluator.evaluate(cv_predictions)
            # print(f"{model_name} (Cross-validated) - Accuracy: {acc}")

    def ML(self):
        self.ML_load()
        self.preprocess_data()
        self.train_test_split()
        self.train_models()


if __name__ == "__main__":
    d = DDA()
    d.ML()
