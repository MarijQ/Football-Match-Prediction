# Plan
# 1. Imports
# 2. File imports and merging
# 3. Feature engineering
# 4. Data cleaning
# 5. EDA
# 6. Machine learning

# 1. Imports
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import random
import os
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col, when, year
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import IndexToString

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)


class DDA():
    def __init__(self):
        pass

    def load_inputs(self):
        self.df_games = pd.read_csv('../1. Input/games.csv')
        self.df_clubs = pd.read_csv('../1. Input/clubs.csv')
        self.df_comps = pd.read_csv('../1. Input/competitions.csv')
        self.df_events = pd.read_csv('../1. Input/game_events.csv')
        self.df_players = pd.read_csv('../1. Input/players.csv')
        self.df_apps = pd.read_csv('../1. Input/appearances.csv')
        self.df_vals = pd.read_csv('../1. Input/player_valuations.csv')

    def df_diagnostics(self):
        print("\n", "Dataframe diagnostics", "\n")
        self.input_dfs = [self.df_games, self.df_clubs, self.df_comps, self.df_events,
                          self.df_players, self.df_apps, self.df_vals]
        for df in self.input_dfs:
            df_name = [k for k, v in vars(self).items() if v is df][0]
            print(f"{df_name}: {df.shape}")
            print("\n", df.isnull().sum(), "\n")

    def subset(self):
        self.df_games = self.df_games[
            ["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id",
             "home_club_goals", "away_club_goals", "home_club_position", "away_club_position",
             "stadium", "attendance", "home_club_formation", "away_club_formation",
             "home_club_name", "away_club_name", "competition_type"]]
        self.df_clubs = self.df_clubs[
            ["club_id", "average_age", "net_transfer_record", "stadium_seats"]]
        self.df_comps = self.df_comps[["competition_id", "name", "country_name"]]
        self.df_events = self.df_events[["game_id", "type", "club_id", "description"]]
        self.df_apps = self.df_apps[["game_id", "player_id", "player_club_id", "assists", "date"]]
        self.df_players = self.df_players[["player_id", "date_of_birth", "height_in_cm"]]
        self.df_vals = self.df_vals[["player_id", "date", "market_value_in_eur"]]

    def filter(self):
        # filter for competition type = domestic league
        self.df_games = self.df_games[self.df_games['competition_type'] == "domestic_league"]
        self.df_apps = self.df_apps[self.df_apps["game_id"].isin(self.df_games["game_id"])]
        self.df_comps = self.df_comps[
            self.df_comps["competition_id"].isin(self.df_games["competition_id"])]
        self.df_events = self.df_events[self.df_events["game_id"].isin(self.df_games["game_id"])]
        self.df_clubs = self.df_clubs[
            self.df_clubs["club_id"].isin(self.df_games["home_club_id"]) |
            self.df_clubs["club_id"].isin(self.df_games["away_club_id"])]
        self.df_players = self.df_players[
            self.df_players["player_id"].isin(self.df_apps["player_id"])]
        self.df_vals = self.df_vals[self.df_vals["player_id"].isin(self.df_apps["player_id"])]
        # filter for games for which there is appearances data
        self.df_games = self.df_games[self.df_games["game_id"].isin(self.df_apps["game_id"])]

    def merge(self):
        # players > appearances
        self.df_apps = pd.merge(self.df_apps, self.df_players, on="player_id", how="left")
        # valuations > appearances
        ## merge market value into appearances (taking latest prior valuation)
        self.df_apps['date'] = pd.to_datetime(self.df_apps['date'])
        self.df_vals['date'] = pd.to_datetime(self.df_vals['date'])
        self.df_apps['app_year'] = self.df_apps['date'].dt.year
        self.df_vals['val_year'] = self.df_vals['date'].dt.year
        x = self.df_vals.groupby(['player_id', 'val_year'], as_index=False)[
            'market_value_in_eur'].mean().reset_index
        # to finish
        # appearances > games
        ## calculate age from DOB and date
        ## merge using avg(height), avg(age), sum(market value) for home/away
        # comps > games
        ## straightforward
        # clubs > games
        ## straightforward
        # events > games
        ## calculate new columns for red/yellow card. Groupby game_id and club_id
        ## Merge into games for both home/away teams using sum()
        pass

    def clean(self):
        # clean remaining cols in master df. Null values / Duplicates / Datatypes
        pass

    def preparation(self):
        self.load_inputs()
        self.subset()
        self.filter()
        self.df_diagnostics()
        self.merge()
        # self.df_diagnostics()

    def EDA(self):
        pass

    def features(self):
        # feature engineering
        ## Occupancy rate calculation
        ## Formation clustering (manual, K-means, comparison)
        ## Country grouping of "other"
        ## One hot encoding for categoricals
        pass

    def ML_load(self):
        self.df = pd.read_csv('../4. Outputs/master_cleaned_csv.csv')
        self.df = self.df[
            ["date", "home_club_position", "away_club_position", "attendance", "home_club_strategy",
             "away_club_strategy", "stadium_seats", "country_name", "home_club_average_age",
             "away_club_average_age", "home_club_average_height", "away_club_average_height",
             "home_club_average_market_value", "away_club_average_market_value", "home_club_goals",
             "away_club_goals"]].head(100)
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

        # Identify categorical columns for one-hot encoding
        categorical_cols = ['home_club_strategy', 'away_club_strategy', 'country_name']
        stages = []

        # StringIndexer and OneHotEncoder for each categorical column
        for col_name in categorical_cols:
            indexer = StringIndexer(inputCol=col_name, outputCol=col_name + '_Index')
            encoder = OneHotEncoder(inputCols=[indexer.getOutputCol()],
                                    outputCols=[col_name + '_Vec'])
            stages += [indexer, encoder]

        # Assemble vectors
        numerical_cols = ['year', 'home_club_position', 'away_club_position', 'attendance',
                          'stadium_seats', 'home_club_average_age', 'away_club_average_age',
                          'home_club_average_height', 'away_club_average_height',
                          'home_club_average_market_value', 'away_club_average_market_value']
        assembler_inputs = [c + '_Vec' for c in categorical_cols] + numerical_cols
        assembler = VectorAssembler(inputCols=assembler_inputs, outputCol='features')
        stages += [assembler]

        # Create a pipeline
        pipeline = Pipeline(stages=stages)
        pipeline_model = pipeline.fit(self.df)
        self.df = pipeline_model.transform(self.df)

    def train_test_split(self):
        # Since we are doing regression, we need two sets of splits, one for each target variable
        # First, we'll predict 'home_club_goals'
        self.train_df_home, self.test_df_home = self.df.selectExpr('features',
                                                                   'home_club_goals as label').randomSplit(
            [0.8, 0.2], seed=42)
        # Then, we'll predict 'away_club_goals'
        self.train_df_away, self.test_df_away = self.df.selectExpr('features',
                                                                   'away_club_goals as label').randomSplit(
            [0.8, 0.2], seed=42)

    def train_models(self):
        # For regression, we will use the RegressionEvaluator
        reg_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction",
                                            metricName="rmse")

        # We train and predict two separate sets of models, one for home club goals and one for away club goals
        for target_variable in ['home', 'away']:
            if target_variable == 'home':
                train_df = self.train_df_home
                test_df = self.test_df_home
            else:
                train_df = self.train_df_away
                test_df = self.test_df_away

            # Linear Regression
            lr = LinearRegression(featuresCol='features', labelCol='label')
            lr_model = lr.fit(train_df)
            lr_predictions = lr_model.transform(test_df)

            # Decision Tree Regressor
            dt = DecisionTreeRegressor(featuresCol='features', labelCol='label')
            dt_model = dt.fit(train_df)
            dt_predictions = dt_model.transform(test_df)

            # Random Forest Regressor
            rf = RandomForestRegressor(featuresCol='features', labelCol='label')
            rf_model = rf.fit(train_df)
            rf_predictions = rf_model.transform(test_df)

            # GBT Regressor
            gbt = GBTRegressor(featuresCol='features', labelCol='label')
            gbt_model = gbt.fit(train_df)
            gbt_predictions = gbt_model.transform(test_df)

            # Evaluate models
            models = {
                'Linear Regression': lr_predictions,
                'Decision Tree': dt_predictions,
                'Random Forest': rf_predictions,
                'GBT': gbt_predictions
            }

            print(f"Results for predicting {target_variable.upper()} Club Goals:")
            for model_name, predictions in models.items():
                rmse = reg_evaluator.evaluate(predictions)
                print(f"{model_name} - RMSE: {rmse}")

            # Uncomment the following lines for additional metrics like MAE or R2
            mae_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
            r2_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
            for model_name, predictions in models.items():
                mae = mae_evaluator.evaluate(predictions)
                r2 = r2_evaluator.evaluate(predictions)
                print(f"{model_name} - MAE: {mae}, R2: {r2}")

    def ML(self):
        self.spark = SparkSession.builder.appName("FootballPredictions").getOrCreate()
        self.ML_load()
        self.preprocess_data()
        self.train_test_split()
        self.train_models()

    def full(self):
        pass


if __name__ == "__main__":
    d = DDA()
    d.ML()
