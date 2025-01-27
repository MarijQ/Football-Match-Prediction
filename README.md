# Football Match Outcome Prediction

Welcome to the GitHub repository for the **Football Match Outcome Prediction** project. This repository showcases a data-driven approach to predict home and away goals in domestic league football matches, based on historical team, player, and match statistics. Created using distributed and non-distributed pipelines, this project compares various regression models and analyzes their performance.

---

## 🚀 Project Overview

### Goal
To predict the number of home and away goals in domestic league football matches using a dataset from **Transfermarkt** that includes match statistics, player valuations, and related metrics from 2012 onward. 

### Key Highlights
- **Distributed Data Processing**: Leveraged PySpark and Hadoop ecosystems for scalable and efficient computational solutions.
- **Multifaceted Machine Learning Pipeline**: Integrated Linear Regression, Decision Tree, and Random Forest models.
- **Exploratory Data Analysis**: Identified key predictive features such as player market values, team formations, and attendance statistics.
- **Feature Engineering**: Created advanced features like occupancy rate and clustered team strategies using K-means for deeper insights.

---

## 👩‍💻 Team

- This project was completed in collaboration with a team of **6 members** over a span of **3 months**.
- **Role**: I served as **team leader**, managing workflows, restructuring the codebase into reusable classes and functions, and overseeing the design of the machine learning pipeline.

---

## 📊 Dataset Overview

### Source
- **Transfermarkt Dataset** (acquired via Kaggle): Includes records of domestic football matches, player valuations, game-specific attributes, and player characteristics.

### Dataset Breakdown
Eight linked relational datasets were used:
- **Games.csv (66k rows)**: Each row represents a football match.
- **Players.csv (30k rows)**: Includes player attributes like age and height.
- **Club_games.csv**: Aggregates team performance in specific games.
- **Player_valuations.csv**: Highlights player market values over time.
- **And more...**

Key research question:  
**Can we predict the number of goals scored by home and away teams using team/player attributes and in-game dynamics?**

---

## 🏗️ Project Structure

```plaintext
combined/
│
├── DDA.pdf                  # Report summarizing the methodology and findings
├── Marij analysis full v3.5.py  # Python code for distributed computations and ML pipelines
├── marij_EDA.py             # EDA and visualization scripts
├── 1. Input/                # Input data directory
│   ├── clubs.csv
│   ├── competitions.csv
│   ├── games.csv
│   ├── player_valuations.csv
│   └── ...
└── data_after_features.csv  # Post-feature-engineering dataset
```

---

## 🔍 Methodology

### 1. Data Preparation
- Filtered datasets for only domestic league games.
- Merged and cleaned attributes across 8 datasets into a single master dataframe.
- Addressed missing values using imputation strategies such as median, mean, and constant values.

### 2. Exploratory Data Analysis
- Conducted statistical and graphical analyses:
  - Correlation matrices to study feature relationships.
  - Visualized trends in team formations, attendance, etc.
- Insights:
  - Player valuations and home occupancy were among the most predictive features for goals.
  - Formations were clustered into **Defensive**, **Offensive**, and **Aggressive** strategies.

### 3. Feature Engineering
- **Occupancy Rate**: Attendance / Stadium Seating Capacity.
- **Team Strategy**: Grouped formations into clusters using K-means.

### 4. Machine Learning
- Preprocessed features using PySpark MLlib for handling categorical variables and vectorized inputs.
- Trained multiple regression models to predict home and away goals:
  - **Models**: Linear Regression, Decision Tree Regressor, and Random Forest.
  - Evaluated models using RMSE, MAE, and R² metrics.

---

## 📊 Key Outcomes

### Model Performance
- **Random Forest** performed best, with:
  - Home Goals RMSE: **1.24**
  - Away Goals RMSE: **1.10**
- However, the R² values remained low (≈0.1), highlighting the complexity of football outcomes.

### Distributed vs Non-Distributed Performance
- Distributed PySpark pipeline scaled effectively to large datasets, outperforming non-distributed methods for datasets >10k rows.

---

## ⚙️ Technology Stack

### Tools
- Python (Pandas, Scikit-learn, Matplotlib, Seaborn)
- PySpark + Hadoop
- K-Means for clustering
- Visualizations using Matplotlib and Seaborn

### Infrastructure
- Linux VM with 12 cores and 32GB RAM for distributed processing.
- Deployed via SSH from a Windows development terminal for seamless file synchronization.

---

## 🌟 Skills Demonstrated
- Team Leadership: Coordinated a team of 5 while also spearheading complex transformations during preprocessing and modeling.
- Big Data Expertise: Implemented distributed computing to process large datasets efficiently in PySpark.
- Machine Learning: Designed robust regression pipelines with hyperparameter tuning.
- Data Engineering: Managed data cleaning, feature engineering, and merging across highly linked relational datasets.

---

## 🛠️ How to Run

1. **Setup Environment**:
   - Install dependencies: `pip install pandas pyspark matplotlib seaborn scikit-learn`.
   - Configure PySpark: Ensure Hadoop and JDK are installed on your system.

2. **Run Preparation & ML Pipeline**:
   ```bash
   python Marij\ analysis\ full\ v3.5.py
   ```

3. **View Results**:
   - Generated output metrics (e.g., RMSE, R²) will be logged.
   - Visualizations can be found in `EDA scripts` or generated live.

---

## 🌏 Future Work
- Incorporate real-time game/event data for enhanced feature sets.
- Explore neural networks or ensemble approaches to improve R².
- Scale to global datasets to analyze league-wide match trends.

---

Thank you for exploring this project! 🚀  
Contributors: **Marij Qureshi & Team**  
Feel free to reach out for questions or collaboration opportunities.
