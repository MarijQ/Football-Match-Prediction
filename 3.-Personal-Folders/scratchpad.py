import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame
x = ["a", "b", "a", "c", "d"]
df = pd.DataFrame(x, columns=["test"])

# Convert 'test' column to a categorical type with an order
df['test'] = pd.Categorical(df['test'], categories=['a', 'b', 'c', 'd'], ordered=True)

# Plotting with seaborn countplot
sns.countplot(data=df, x='test', order=['a', 'b', 'c', 'd'])
plt.show()