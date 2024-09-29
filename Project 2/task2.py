# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Iris dataset
iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(iris_url, header=None, names=column_names)

# Explore the structure of the dataset
print("Data Structure:")
print(iris_data.info())
print("\nFirst 5 rows of the dataset:")
print(iris_data.head())

# 2. Check for missing values
missing_values = iris_data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Handle missing values if necessary (In this case, there are no missing values in the Iris dataset)

# 3. Visualize the data
# Set the style of the visualization
sns.set(style='whitegrid')

# Pairplot to visualize relationships between features
sns.pairplot(iris_data, hue='species')
plt.title('Pairplot of Iris Dataset')
plt.show()

# Boxplot to understand the distribution of each feature
plt.figure(figsize=(12, 8))
for i, column in enumerate(column_names[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=column, data=iris_data)
    plt.title(f'Boxplot of {column}')

plt.tight_layout()
plt.show()

# Histogram for each feature
iris_data.hist(figsize=(10, 8), bins=20, edgecolor='black')
plt.suptitle('Histograms of Iris Dataset Features')
plt.show()