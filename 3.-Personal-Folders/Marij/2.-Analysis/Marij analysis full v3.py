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
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import stddev
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
import time

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

    def master_df_diagnostics(self):
        print(f"\nshape: {self.df.shape}\n")
        print(self.df.head(5))
        print(f"\n\n{self.df.info()}\n")
        print(f"\nnull values:\n\n{self.df.isnull().sum()}\n")

    def subset(self):
        self.df_games = self.df_games[
            ["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals", "home_club_position", "away_club_position", "attendance",
             "home_club_formation", "away_club_formation", "competition_type"]]
        self.df_clubs = self.df_clubs[["club_id", "net_transfer_record", "stadium_seats"]]
        self.df_comps = self.df_comps[["competition_id", "name", "country_name"]]
        self.df_events = self.df_events[["game_id", "type", "club_id", "description"]]
        self.df_apps = self.df_apps[["game_id", "player_id", "player_club_id", "assists", "date"]]
        self.df_players = self.df_players[["player_id", "date_of_birth", "height_in_cm"]]
        self.df_vals = self.df_vals[["player_id", "date", "market_value_in_eur"]]

    def filter(self):
        # filter for competition type = domestic league
        self.df_games = self.df_games[self.df_games['competition_type'] == "domestic_league"]
        # propagate game_id filter across all other datasets
        self.df_apps = self.df_apps[self.df_apps["game_id"].isin(self.df_games["game_id"])]
        self.df_comps = self.df_comps[self.df_comps["competition_id"].isin(self.df_games["competition_id"])]
        self.df_events = self.df_events[self.df_events["game_id"].isin(self.df_games["game_id"])]
        self.df_clubs = self.df_clubs[self.df_clubs["club_id"].isin(self.df_games["home_club_id"]) | self.df_clubs["club_id"].isin(self.df_games["away_club_id"])]
        self.df_players = self.df_players[self.df_players["player_id"].isin(self.df_apps["player_id"])]
        self.df_vals = self.df_vals[self.df_vals["player_id"].isin(self.df_apps["player_id"])]
        # filter for games for which there is appearances data
        self.df_games = self.df_games[self.df_games["game_id"].isin(self.df_apps["game_id"])]

    def merge(self):
        # players > appearances
        self.df_apps = pd.merge(self.df_apps, self.df_players, on="player_id", how="left")

        # valuations > appearances
        self.df_apps['date'] = pd.to_datetime(self.df_apps['date'])
        self.df_vals['date'] = pd.to_datetime(self.df_vals['date'])
        self.df_apps = self.df_apps.sort_values(by='date')
        self.df_vals = self.df_vals.sort_values(by='date')
        self.df_apps = pd.merge_asof(self.df_apps, self.df_vals, by='player_id', on='date', direction='backward')

        # appearances > games
        self.df_apps['age'] = (self.df_apps['date'] - pd.to_datetime(self.df_apps['date_of_birth'])).dt.days // 365
        apps_agg = self.df_apps.groupby(['game_id', 'player_club_id']).agg({'age': 'mean', 'height_in_cm': 'mean', 'market_value_in_eur': 'mean', 'assists': 'sum'}).reset_index()

        # Make sure that player_club_id is either home_club_id or away_club_id, no need to carry it after merge
        apps_home_agg = apps_agg.rename(columns={"player_club_id": "home_club_id"})
        apps_away_agg = apps_agg.rename(columns={"player_club_id": "away_club_id"})

        self.df_games = pd.merge(self.df_games, apps_home_agg, on=['game_id', 'home_club_id'], how='left')
        self.df_games = pd.merge(self.df_games, apps_away_agg, on=['game_id', 'away_club_id'], how='left', suffixes=('_home', '_away'))

        # comps > games
        self.df_games = pd.merge(self.df_games, self.df_comps, on='competition_id', how='left')

        # clubs > games
        # We will add suffixes _home and _away during merge since club_id exists in both
        self.df_games = pd.merge(self.df_games, self.df_clubs.add_suffix('_home'), left_on='home_club_id', right_on='club_id_home', how='left')
        self.df_games = pd.merge(self.df_games, self.df_clubs.add_suffix('_away'), left_on='away_club_id', right_on='club_id_away', how='left')

        # events > games
        self.df_events['red_card'] = self.df_events['description'].str.contains('red card', case=False, na=False).astype(int)
        self.df_events['yellow_card'] = self.df_events['description'].str.contains('yellow card', case=False, na=False).astype(int)
        events_agg = self.df_events.groupby(['game_id', 'club_id']).agg({'red_card': 'sum', 'yellow_card': 'sum'}).reset_index()

        # Ensure correct suffixes and rename for clarity
        events_home_agg = events_agg.rename(columns={"club_id": "home_club_id", "red_card": "home_red_cards", "yellow_card": "home_yellow_cards"})
        events_away_agg = events_agg.rename(columns={"club_id": "away_club_id", "red_card": "away_red_cards", "yellow_card": "away_yellow_cards"})

        self.df_games = pd.merge(self.df_games, events_home_agg, on=['game_id', 'home_club_id'], how='left')
        self.df_games = pd.merge(self.df_games, events_away_agg, on=['game_id', 'away_club_id'], how='left')

        # After merging, we can drop merged club_ids since they are already represented by home_club_id and away_club_id
        self.df_games.drop(columns=['club_id_home', 'club_id_away'], inplace=True)

        # Define the new master dataframe with only the relevant columns and the desired order
        self.df = self.df_games[
            ['season', 'date', 'home_club_id', 'away_club_id', 'home_club_position', 'away_club_position', 'home_club_formation', 'away_club_formation', 'age_home', 'age_away', 'height_in_cm_home',
             'height_in_cm_away',
             'market_value_in_eur_home', 'market_value_in_eur_away', 'assists_home', 'assists_away', 'attendance', 'home_red_cards', 'away_red_cards', 'home_yellow_cards',
             'away_yellow_cards', 'name', 'country_name', 'net_transfer_record_home', 'net_transfer_record_away', 'stadium_seats_home', 'home_club_goals', 'away_club_goals']]

        # Rename the columns for clarity
        self.df.columns = ['year', 'date', 'home_id', 'away_id', 'pos_home', 'pos_away', 'formation_home', 'formation_away', 'age_home', 'age_away', 'height_home', 'height_away', 'val_home',
                           'val_away', 'assists_home',
                           'assists_away', 'attendance', 'red_home', 'red_away', 'yellow_home', 'yellow_away', 'league', 'country', 'ntr_home', 'ntr_away', 'seats', 'goals_home', 'goals_away']

    def clean(self):
        def drop_rows_with_nulls(subset):
            self.df.dropna(subset=subset, inplace=True)

        def impute_value(columns, method, value=None, groupby_attributes=None):
            """Impute missing values in the specified columns.
            :param columns: List of column names to impute.
            :param method: Imputation method ('mean', 'median', 'mode', 'constant').
            :param value: Constant value for imputation if method is 'constant'.
            :param groupby_attributes: List of columns to group by for group-wise imputation.
            """
            for column in columns:
                if method == 'constant' and value is not None:
                    # Impute with a constant value
                    self.df.loc[:, column] = self.df[column].fillna(value)
                elif groupby_attributes:
                    # Group-wise imputation
                    if method == 'mode':
                        # 'mode' requires a custom lambda function with groupby
                        mode_series = self.df.groupby(groupby_attributes)[column].apply(
                            lambda x: x.mode()[0] if not x.mode().empty else x)
                        self.df.loc[:, column] = self.df[column].fillna(mode_series)
                    else:
                        # Standard mean or median can be used with transform
                        self.df.loc[:, column] = self.df.groupby(groupby_attributes)[column].transform(method).fillna(self.df[column])
                else:
                    # Global column-wise imputation
                    self.df.loc[:, column] = self.df[column].fillna(self.df[column].agg(method))

        # Remove duplicates
        self.df = self.df.drop_duplicates()

        # Correct strings for currency conversion
        currency_columns = ['ntr_home', 'ntr_away']
        for col in currency_columns:
            self.df[col] = (self.df[col].str.replace('€', '', regex=False).str.replace('m', 'e6', regex=False).str.replace('k', 'e3', regex=False)
                            .str.replace('+', '', regex=False).str.replace('−', '-', regex=False).astype(float))

        # Correct datatypes
        for col in ['formation_home', 'formation_away', 'league', 'country']:
            self.df[col] = self.df[col].astype('category')

        self.df['seats'] = pd.to_numeric(self.df['seats'], errors='coerce')
        numeric_cols = ['pos_home', 'pos_away', 'age_home', 'age_away', 'height_home', 'height_away', 'val_home', 'val_away', 'assists_home', 'assists_away', 'attendance', 'red_home',
                        'red_away', 'yellow_home', 'yellow_away']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Closed-door matches in 2020 due to COVID-19 pandemic
        self.df.loc[(self.df['year'] == 2020), 'attendance'] = self.df.loc[(self.df['year'] == 2020), 'attendance'].fillna(0)

        # Set attendance to zero for Ukrainian and Russian clubs for the 2022 season onwards due to the conflict
        self.df.loc[((self.df['country'] == 'Ukraine') | (self.df['country'] == 'Russia')) & (self.df['year'] >= 2022), 'attendance'] = 0

        # For remaining null 'attendance' values, impute using mean of 'league' and 'home_id'
        impute_value(['attendance'], method='mean', groupby_attributes=['league', 'home_id'])

        # Market values could vary between leagues; mean is grouped by 'league' and 'home_id'
        impute_value(['val_home', 'val_away'], method='mean', groupby_attributes=['league', 'home_id'])

        # Team positions change annually and can differ by league
        impute_value(['pos_home', 'pos_away'], method='median', groupby_attributes=['year', 'league'])

        # Impute remaining columns with potential zero/null default values
        impute_value(['red_home', 'red_away', 'yellow_home', 'yellow_away'], method='constant', value=0)
        impute_value(['assists_home', 'assists_away'], method='constant', value=0)

        # Drop rows with missing ages as they are crucial and cannot be accurately imputed
        drop_rows_with_nulls(['age_home', 'age_away'])

        # Impute heights using median grouped by team since players' heights are consistent within a team
        impute_value(['height_home', 'height_away'], method='median', groupby_attributes=['home_id'])

        # Impute formations using the mode for each team
        impute_value(['formation_home'], method='mode', groupby_attributes=['home_id'])
        impute_value(['formation_away'], method='mode', groupby_attributes=['away_id'])

    def preparation(self):
        self.load_inputs()
        self.subset()
        self.filter()
        self.merge()
        self.clean()
        # self.master_df_diagnostics()

    def EDA_table(self):
        country_league_table = pd.crosstab(self.df['country'], self.df['league'], margins=False)
        print(country_league_table)

    def EDA(self):
        # seaborn heatmap with labeled x-axis categories
        numeric_vars = ['year', 'pos_home', 'pos_away', 'age_home', 'age_away',
                        'height_home', 'height_away', 'val_home', 'val_away',
                        'assists_home', 'assists_away', 'red_home', 'red_away',
                        'yellow_home', 'yellow_away', 'ntr_home', 'ntr_away',
                        'seats', 'goals_home', 'goals_away', 'attendance']

        correlation_matrix = self.df[numeric_vars].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, square=True, cmap='coolwarm', annot=True, fmt='.2f')
        plt.xticks(rotation='vertical')
        plt.title('Correlation Matrix of Numerical Variables')
        plt.show()

        # Table of game counts with rows for country and columns for league
        country_league_table = pd.crosstab(self.df['country'], self.df['league'], margins=False)
        print(country_league_table)

        # Horizontal bar chart for frequency of home and away formations, ranked by frequency
        formation_order_home = self.df['formation_home'].value_counts().index
        sns.countplot(data=self.df, y='formation_home', order=formation_order_home)
        plt.title('Frequency of Different Home Formations')
        plt.xlabel('Frequency')
        plt.ylabel('Home Formation')
        plt.show()

        formation_order_away = self.df['formation_away'].value_counts().index
        sns.countplot(data=self.df, y='formation_away', order=formation_order_away)
        plt.title('Frequency of Different Away Formations')
        plt.xlabel('Frequency')
        plt.ylabel('Away Formation')
        plt.show()

    def features(self):
        def occupancy_rate_calculation():
            self.df['occupancy'] = self.df['attendance'] / self.df['seats']
            self.df.drop(columns=['attendance', 'seats'], inplace=True)

        def formation_clustering():
            # First, create a combined list of all formations as strings
            all_formations = self.df['formation_home'].astype(str).tolist() + self.df['formation_away'].astype(str).tolist()

            # Helper function to preprocess the formation strings into numeric values
            def preprocess_formation(formation):
                # Convert the formation string into a list of defenders, midfielders, attackers
                nums = [int(n) for n in formation.split('-') if n.isdigit()]
                if len(nums) == 3:
                    return nums
                elif len(nums) == 4:
                    return [nums[0], nums[1] + nums[2], nums[3]]
                return [4, 4, 2]  # Default formation

            # Preprocess all formations
            processed_formations = [preprocess_formation(f) for f in all_formations if f != 'nan']

            # Perform the clustering on numeric formation data
            scaler = StandardScaler()
            formation_scaled = scaler.fit_transform(processed_formations)

            # Run the clustering algorithm
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(formation_scaled)

            # Define a function to get the label for a formation
            def get_formation_label(formation):
                # Predict the cluster for the formation and return its label
                formation_numeric = preprocess_formation(formation)
                formation_scaled = scaler.transform([formation_numeric])
                return kmeans.predict(formation_scaled)[0]

            # Label each formation with the cluster number
            self.df['strategy_home'] = self.df['formation_home'].astype(str).apply(get_formation_label)
            self.df['strategy_away'] = self.df['formation_away'].astype(str).apply(get_formation_label)

            # Print the cluster centers
            # print("Cluster centers:")
            # print(kmeans.cluster_centers_)

            # Define your interpretations for each cluster
            cluster_interpretations = {0: "Defensive", 1: "Offensive", 2: "Aggressive"}

            # Apply the interpretations to the dataframe
            self.df['strategy_home'] = self.df['strategy_home'].map(cluster_interpretations)
            self.df['strategy_away'] = self.df['strategy_away'].map(cluster_interpretations)

            # Dropping the old formation columns after assigning strategies
            self.df.drop(columns=['formation_home', 'formation_away'], inplace=True)

        # Execute the nested functions
        occupancy_rate_calculation()
        formation_clustering()

        # # drop ids and rename country_group
        # self.df.drop(columns=['home_id', 'away_id'], inplace=True)

    def EDA_2(self):
        # Bar chart of year with year as x-axis
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.df, x='year')
        plt.title('Distribution of Year')
        plt.xlabel('Year')
        plt.ylabel('Frequency')
        plt.show()

        # Histogram of occupancy rate with KDE
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df['occupancy'], bins=30, kde=True)
        plt.title('Distribution of Occupancy Rate')
        plt.xlabel('Occupancy Rate')
        plt.ylabel('Frequency')
        plt.show()

        # Dealing with occupancy rate greater than 1 by setting it to 1
        self.df['occupancy'] = self.df['occupancy'].clip(upper=1)

        # Frequency bar chart of country_group, ordered by frequency, with labeled x-axis
        country_group_order = self.df['country_group'].value_counts().index
        sns.countplot(data=self.df, x='country_group', order=country_group_order)
        plt.title('Frequency of Country Groups')
        plt.xlabel('Country Group')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()

        # Box plot of home goals by home strategy with text label for x-axis
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='strategy_home', y='goals_home')
        plt.title('Home Goals by Home Strategy')
        plt.xlabel('Home Strategy')
        plt.ylabel('Home Goals')
        plt.xticks(rotation=45)
        plt.show()

        # Box plot of away goals by away strategy with text label for x-axis
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='strategy_away', y='goals_away')
        plt.title('Away Goals by Away Strategy')
        plt.xlabel('Away Strategy')
        plt.ylabel('Away Goals')
        plt.xticks(rotation=45)
        plt.show()

    def exploration(self):
        self.df.to_csv('./data_after_EDA.csv', index=False)
        self.EDA_table()
        # self.EDA()
        self.features()
        self.df.to_csv('./data_after_features.csv', index=False)
        # self.EDA_2()
        # self.master_df_diagnostics()

    def ML_load(self):
        self.spark = SparkSession.builder.appName("DistributedDataAnalysis").getOrCreate()
        self.df = self.spark.read.csv('./data_after_features.csv', header=True, inferSchema=True)
        # Sample X% of the data randomly for speed comparison
        # self.df = self.df.sample(fraction=0.1, withReplacement=False, seed=42)
        self.df = self.df.drop('date', 'home_id', 'away_id', 'league', 'assists_home', 'assists_away')
        self.df = self.df.withColumnRenamed('avg_val_home', 'val_home').withColumnRenamed('avg_val_away', 'val_away')

    def preprocess_data(self):
        # Identify categorical features and apply StringIndexer
        categorical_cols = ['country', 'strategy_home', 'strategy_away']
        indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_index").fit(self.df)
            for col in categorical_cols
        ]

        # Apply OneHotEncoder to the indexed categorical features
        encoders = [
            OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=f"{col}_vec")
            for col, indexer in zip(categorical_cols, indexers)
        ]

        # Assemble new feature columns—numeric features, plus the new one-hot encoded variables
        self.numeric_cols = [col for col in self.df.columns if
                             col not in ['country', 'strategy_home', 'strategy_away', 'goals_home', 'goals_away', 'date', 'home_id', 'away_id', 'league', 'assists_home', 'assists_away']]
        self.feature_cols_home = self.numeric_cols + ['country_vec', 'strategy_home_vec']
        self.feature_cols_away = self.numeric_cols + ['country_vec', 'strategy_away_vec']

        self.assembler_home = VectorAssembler(inputCols=self.feature_cols_home, outputCol="features")
        self.assembler_away = VectorAssembler(inputCols=self.feature_cols_away, outputCol="features")

        self.pipeline_home = Pipeline(stages=indexers + encoders + [self.assembler_home])
        self.pipeline_away = Pipeline(stages=indexers + encoders + [self.assembler_away])

    def train_test_split(self):
        # Split the data into train and test sets for both home and away models
        self.train_data_home, self.test_data_home = self.df.randomSplit([0.8, 0.2], seed=42)
        self.train_data_away, self.test_data_away = self.df.randomSplit([0.8, 0.2], seed=42)

    def train_models(self):
        # Define models
        lr = LinearRegression(maxIter=5)
        dt = DecisionTreeRegressor()
        rf = RandomForestRegressor(numTrees=10)

        # Define parameter grids for each model
        paramGrid_lr = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.1, 0.01]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()
        paramGrid_dt = ParamGridBuilder() \
            .addGrid(dt.maxDepth, [2, 5, 10]) \
            .build()
        paramGrid_rf = ParamGridBuilder() \
            .addGrid(rf.maxDepth, [5, 10]) \
            .addGrid(rf.maxBins, [32, 64]) \
            .build()

        # Define evaluators
        evaluator_rmse = RegressionEvaluator(metricName="rmse")
        evaluator_mae = RegressionEvaluator(metricName="mae")
        evaluator_r2 = RegressionEvaluator(metricName="r2")

        # Set up CrossValidators
        crossval_lr = CrossValidator(estimator=lr,
                                     estimatorParamMaps=paramGrid_lr,
                                     evaluator=evaluator_rmse,
                                     numFolds=3)
        crossval_dt = CrossValidator(estimator=dt,
                                     estimatorParamMaps=paramGrid_dt,
                                     evaluator=evaluator_rmse,
                                     numFolds=3)
        crossval_rf = CrossValidator(estimator=rf,
                                     estimatorParamMaps=paramGrid_rf,
                                     evaluator=evaluator_rmse,
                                     numFolds=3)

        models_and_params = {
            'LinearRegression': crossval_lr,
            'DecisionTree': crossval_dt,
            'RandomForest': crossval_rf
        }

        # A utility function to extract feature names from VectorAssembler
        def extract_feature_names(pipelineModel, inputCols):
            feature_names = []
            for i, stage in enumerate(pipelineModel.stages):
                if isinstance(stage, VectorAssembler):
                    feature_names.extend(stage.getInputCols())
                elif isinstance(stage, OneHotEncoder):
                    category_col = stage.getInputCols()[0].replace('_index', '')
                    # For each category, derive the feature names based on the number of categories present
                    ohe_cols_count = len(stage.categorySizes)
                    feature_names.extend([f'{category_col}_{i}' for i in range(ohe_cols_count)])
            return feature_names

        for goal in ('home', 'away'):
            # Train and evaluate models for home and away goals
            for model_name, crossval in models_and_params.items():
                print(f"\nTraining {model_name} model for {goal} goals prediction.")

                # Train the model
                pipeline = self.pipeline_home if goal == "home" else self.pipeline_away
                train_data = self.train_data_home if goal == "home" else self.train_data_away
                test_data = self.test_data_home if goal == "home" else self.test_data_away
                train_data = train_data.withColumnRenamed(f"goals_{goal}", "label")
                test_data = test_data.withColumnRenamed(f"goals_{goal}", "label")

                model_pipeline = pipeline.fit(train_data)
                cvModel = crossval.fit(model_pipeline.transform(train_data))

                # Evaluation
                predictions = cvModel.transform(model_pipeline.transform(test_data))
                rmse = evaluator_rmse.evaluate(predictions)
                mae = evaluator_mae.evaluate(predictions)
                r2 = evaluator_r2.evaluate(predictions)

                # Show metrics
                print(f"{model_name} - RMSE: {rmse}")
                print(f"{model_name} - MAE: {mae}")
                print(f"{model_name} - R^2: {r2}")

                # Feature importance (only works for models that provide this information)
                for model_name, crossval in models_and_params.items():
                    if model_name == 'RandomForest':
                        bestModel = cvModel.bestModel
                        pipelineModel = self.pipeline_home.fit(self.train_data_home) if goal == "home" else self.pipeline_away.fit(self.train_data_away)
                        # Use self.feature_cols_home and self.feature_cols_away since they have been defined
                        feature_cols_home = extract_feature_names(pipelineModel, self.feature_cols_home)
                        feature_cols_away = extract_feature_names(pipelineModel, self.feature_cols_away)
                        inputCols = feature_cols_home if goal == "home" else feature_cols_away
                        featureNames = extract_feature_names(pipelineModel, inputCols)
                        if hasattr(bestModel, 'featureImportances'):
                            # Map feature importance values to feature names
                            featureImportances = bestModel.featureImportances
                            results = [(name, featureImportances[idx]) for idx, name in enumerate(featureNames) if featureImportances[idx]]
                            sortedResults = sorted(results, key=lambda x: x[1], reverse=True)
                            print(f"Feature importances for {goal} goals prediction:")
                            for name, importance in sortedResults:
                                print(f"{name}: {importance}")

        # Calculate benchmarks (standard deviation of the target variables)
        self.benchmark_home = self.df.select(stddev(col(f"goals_home"))).collect()[0][0]
        self.benchmark_away = self.df.select(stddev(col(f"goals_away"))).collect()[0][0]
        print(f"Benchmark Home Goals: {self.benchmark_home:.2f}")
        print(f"Benchmark Away Goals: {self.benchmark_away:.2f}")

    def ML(self):
        self.ML_load()
        self.preprocess_data()
        self.train_test_split()
        self.train_models()

    def ML_non_distributed(self):
        # Load data
        df = pd.read_csv('./data_after_features.csv')
        # Sample X% of the data randomly for speed comparison
        df = df.sample(frac=0.1, random_state=42)
        df = df.drop(['date', 'home_id', 'away_id', 'league', 'assists_home', 'assists_away'], axis=1)
        df.rename(columns={'avg_val_home': 'val_home', 'avg_val_away': 'val_away'}, inplace=True)

        # Define categorical and numeric features
        categorical_features = ['country', 'strategy_home', 'strategy_away']
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features.remove('goals_home')  # Assuming 'goals_home' is the target for this example

        # Define preprocessing for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Combine preprocessors into a single ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Define the model pipelines
        models = [
            ('LinearRegression', LinearRegression()),
            ('DecisionTree', DecisionTreeRegressor(random_state=42)),
            ('RandomForest', RandomForestRegressor(random_state=42))
        ]

        # Train and evaluate models
        for name, model in models:
            # Define pipeline with preprocessing and model
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('model', model)])

            # Split the data
            X = df.drop('goals_home', axis=1)  # Features
            y = df['goals_home']  # Target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Define hyperparameters for grid search based on the model
            if name == 'RandomForest':
                param_grid = {
                    'model__n_estimators': [50, 100],
                    'model__max_features': ['sqrt', 'log2'],
                    'model__max_depth': [10, 20]
                }
            elif name == 'DecisionTree':
                param_grid = {
                    'model__max_depth': [3, 5, 10]
                }
            elif name == 'LinearRegression':
                param_grid = {
                    # Remove 'normalize' from the parameter grid, since it's no longer valid
                    # Update the param_grid if you want to add other parameters for tuning
                }

            # Run grid search
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)

            # Get and evaluate the best model from grid search
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Output model performance metrics
            print(f'{name} - Best Params: {grid_search.best_params_}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}')

            # Feature importance for Tree-based models
            if hasattr(best_model.named_steps['model'], 'feature_importances_'):
                # Retrieve preprocessing pipeline from the best model
                preprocessor_pipeline = best_model.named_steps['preprocessor']

                # Access OneHotEncoder within the preprocessor pipeline
                categorical_transformer = preprocessor_pipeline.named_transformers_['cat']

                # Retrieve the categorical feature names with appropriate method get_feature_names_out
                categorical_feature_names = categorical_transformer.get_feature_names_out()

                # Combine categorical and numeric feature names
                transformed_features = (numeric_features + list(categorical_feature_names))

                # Get feature importances
                importances = best_model.named_steps['model'].feature_importances_

                # Feature importance extraction
                feature_importances = pd.DataFrame(
                    sorted(zip(transformed_features, importances), key=lambda x: x[1], reverse=True),
                    columns=['feature', 'importance']
                )
                print(feature_importances.head())

    def full(self):
        # self.preparation()
        # self.exploration()
        # Start the timer for the non-distributed part
        # start_time_non_dist = time.time()
        # self.ML_non_distributed()
        # # End the timer for the non-distributed part and print the runtime
        # end_time_non_dist = time.time()
        # print(f"Non-distributed ML runtime: {end_time_non_dist - start_time_non_dist} seconds")

        # Start the timer for the distributed part
        start_time_dist = time.time()
        self.ML()
        # End timer for the distributed part and print the runtime
        end_time_dist = time.time()
        print(f"Distributed ML runtime: {end_time_dist - start_time_dist} seconds")


if __name__ == "__main__":
    d = DDA()
    d.full()
