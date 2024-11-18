
from utils.data_understanding_helpers import find_outliers


def drop_outliers(df, columns, threshold=3):
    """
    Drops rows with outliers in the specified columns of a DataFrame using Z-scores.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to calculate Z-scores for.
        threshold (float): The Z-score threshold for identifying outliers (default is 3).

    Returns:
        pd.DataFrame: A DataFrame with outliers removed.
    """
    # Find the outliers using the existing find_outliers function
    outliers_df = find_outliers(df, columns, threshold)
    
    # Drop the rows with outliers from the original DataFrame
    df_without_outliers = df.drop(outliers_df.index)
    
    return df_without_outliers