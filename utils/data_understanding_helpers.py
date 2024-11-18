from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def analyze_missing_values(df):
    """
    Analyze and display missing values in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze for missing values.

    This function calculates the total count and percentage of missing values for each feature (column) in the DataFrame,
    and prints the results in a neatly formatted table using the `tabulate` module.
    """

    # Calculate the total number of missing values per feature
    missing_values = df.isnull().sum()

    # Calculate the percentage of missing values per feature
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    # Create a list to store the results for each feature
    missing_data = []

    # Loop through the missing values and their corresponding percentages
    for feature, count in missing_values.items():
        percentage = missing_percentage[feature]
        # Append the feature, count of missing values, and percentage to the list
        missing_data.append([feature, count, f"{percentage:.2f}%"])

    # Define the headers for the table
    headers = ["Feature", "#Missing", "%Missing"]

    # Print the missing values analysis using `tabulate`
    print("\nMissing Values Analysis:")
    print(tabulate(missing_data, headers=headers, tablefmt="grid"))


def analyze_categorical_features(df):
    """
    Analyze and display categorical features in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze for categorical features.

    This function identifies and prints the categorical features (columns with object data type) in the DataFrame.
    """
    categorical_features = df.select_dtypes(include=['object']).columns
    print("\nCategorical Features Analysis:")
    print("Feature: {}".format(categorical_features))
    return categorical_features



def plot_histograms(data, kde=True):
    """
    Plots histograms with KDE for a list of pandas Series or a single Series.
    
    Parameters:
    data (list or single pandas Series): A pandas Series or a list of Series to plot.
    """
    # Ensure data is a list, even if a single Series is passed
    if not isinstance(data, list):
        data = [data]
    
    # Create a figure with subplots
    num_columns = len(data)
    num_rows = (num_columns // 2) + (num_columns % 2)  # Ensures enough rows for subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, 8))
    axes = axes.flatten()  # Flatten axes array to easily loop over it
    
    for i, series in enumerate(data):
        # Extract the name of the column for the title
        column_name = series.name if series.name else f"Column {i + 1}"
        
        # Plot the histogram with KDE
        sns.histplot(series.dropna(), bins=30, kde=kde, ax=axes[i])
        axes[i].set_title(f"Distribution of {column_name}")
        axes[i].set_xlabel(column_name)  # Correct xlabel to the actual column name
        axes[i].set_ylabel('Frequency')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout for better display
    plt.tight_layout()
    plt.show()



def find_outliers(df, columns, threshold=3):
    """
    Identifies outliers in the specified columns of a DataFrame using Z-scores.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to calculate Z-scores for.
        threshold (float): The Z-score threshold for identifying outliers (default is 3).
        
    Returns:
        pd.DataFrame: A DataFrame containing only the rows that have outliers.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate Z-scores for the specified columns
    for column in columns:
        zscore_column = f'zscore_{column}'
        df_copy[zscore_column] = zscore(df_copy[column])

    # Identify rows where Z-scores are above the threshold
    outliers_condition = (df_copy[[f'zscore_{col}' for col in columns]].abs() > threshold).any(axis=1)
    
    # Return only the rows with outliers
    outliers_df = df_copy[outliers_condition]
    
    return outliers_df

def plot_correlation_matrix(df, data_description):
    """
    Plots a correlation matrix for the passed DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        data_description (str): A description of the data to be used in the plot title.
    """
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Get all numeric columns in a new dataframe
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Create a heatmap of the correlation matrix using seaborn
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)

    # Add title and display the plot
    plt.title(f'Correlation Heatmap of {data_description}')
    plt.tight_layout()
    plt.show()