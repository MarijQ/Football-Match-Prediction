import matplotlib.pyplot as plt
import seaborn as sns
import random
from matplotlib import cm


class EDA:
    def __init__(self, data):
        self.data = data

    # displaying rows
    # "n" describes how many rows to show
    def display(self, n):
        print(self.data.head(n))

    # Summary Statistics
    def stats(self):
        print(self.data.describe())

    # Histograms
    def hist_plot(self, column):
        plt.figure()  # creates figure object to make graph
        sns.histplot(data=self.data[column])
        plt.title(f"Histogram of {column}")
        plt.show()

    # Bar Charts
    def bar_plot(self, x_col, y_col):
        plt.figure()
        color_value = random.random()
        sns.barplot(data=self.data, x=x_col, y=y_col,
                    hue=x_col, palette="Dark2",
                    dodge=False, legend=False)
        plt.title(f"bar chart of {x_col} and {y_col}")
        plt.show()

    # Scatter Plots
    def scat_plot(self, x_col, y_col):
        plt.figure()
        # Generate a random color using RGB values
        color_value = random.random()
        colormap = cm.get_cmap('crest')
        color = colormap(color_value)
        sns.scatterplot(data=self.data, x=x_col, y=y_col, color=color)
        plt.title(f"scatter chart of {x_col} and {y_col}")
        plt.show()

    # count of observations for categorical cols.
    def catplot(self, col_name):
        plt.figure()
        sns.countplot(data=self.data, x=col_name,
                      hue=col_name, palette='viridis')
        plt.title(f'Countplot of {col_name}')
        plt.show()

    # Line Charts
    def line_plot(self, x_col, y_col):
        plt.figure()
        sns.lineplot(data=self.data, x=x_col, y=y_col)
        plt.title(f"line chart of {x_col} and {y_col}")
        plt.show()

    # Box plots
    def box_plot(self, x_col, y_col):
        plt.figure()
        sns.boxplot(data=self.data, x=x_col, y=y_col,
                    hue=x_col, palette='viridis')
        plt.title(f"box chart of {x_col} and {y_col}")
        plt.show()

    # Heatmap
    def mheat(self):

        # select only the column that has number in it.
        corr = self.data.select_dtypes(include=['number'])

        ''' .empty checks if the dataframe (corr) is empty or not. 
        If it is empty,it will print the message. If it is not (if not corr.empty), it will create a heatmap.
        '''
        if not corr.empty:
            plt.figure()
            sns.heatmap(data=corr.corr(), annot=True, cmap='PiYG')
            plt.title("Correlation Heatmap")
            plt.show()
        else:
            print("No numeric data found")

    # Kde plots
    def kde_plot(self, x_col, cat_var):
        plt.figure()
        sns.kdeplot(data=self.data, x=x_col, hue=cat_var, multiple='fill')
        plt.title(f"KDE plot of {x_col} and {cat_var}")
        plt.show()