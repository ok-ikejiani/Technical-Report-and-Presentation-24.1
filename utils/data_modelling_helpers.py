# Standard Library Imports
import copy  # For making deep copies of mutable objects
import time # For timing the training of models
import warnings  # For suppressing warnings
warnings.filterwarnings('ignore')

# Data Manipulation and Visualization Libraries
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Data visualization
import matplotlib.cm as cm  # For color mapping
from tabulate import tabulate  # For printing data in table format

# Scikit-learn: Model Selection, Metrics, and Preprocessing
from sklearn.model_selection import GridSearchCV  # Hyperparameter tuning
from sklearn.dummy import DummyClassifier, DummyRegressor  # Baseline models
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc,  # Classification metrics
                             mean_squared_error, r2_score)  # Regression metrics
from sklearn.preprocessing import PolynomialFeatures  # For generating polynomial features
from sklearn.decomposition import PCA  # For dimensionality reduction

# Imbalanced-learn: Pipelines and Resampling Techniques
from imblearn.pipeline import Pipeline  # Pipeline to include resampling steps
from imblearn.under_sampling import RandomUnderSampler  # For undersampling imbalanced datasets
from imblearn.over_sampling import SMOTE  # For oversampling imbalanced datasets

# Scikit-learn: Model Inspection
from sklearn.inspection import permutation_importance  # Permutation importance for model interpretation

class ModelWrapper:
    def __init__(self, estimator, param_grid, name):
        """
        Class that encapsulates a model, its hyperparameters for grid search, and its name.

        Args:
            model (estimator): The model (e.g., sklearn estimator) to be tuned.
            param_grid (dict): The hyperparameters to use in GridSearchCV.
            name (str): The name of the model.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.name = name


# Define the scoring metric mappings for both model types
SCORING_MAP = {
    'classification': {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    },
    'regression': {
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'mse': 'neg_mean_squared_error',
        'msle': 'neg_mean_squared_log_error'
    }
}

def is_valid_scoring_metric(model_type, scoring_metric):
    """
    Validates the scoring metric for the given model type.
    
    Args:
        model_type: 'classification' or 'regression' to specify the type of model.
        scoring_metric: Metric to evaluate the models.

    Raises:
        ValueError: If the scoring metric is invalid for the provided model type.
    """
    # Convert scoring metric to lowercase
    scoring_metric = scoring_metric.lower()

    # Validate based on model type using the SCORING_MAP
    if model_type in SCORING_MAP:
        if scoring_metric not in SCORING_MAP[model_type]:
            raise ValueError(f"Invalid scoring metric for {model_type}: {scoring_metric}. "
                             f"Choose from: {list(SCORING_MAP[model_type].keys())}")
    else:
        raise ValueError("Invalid model type provided. Use 'classification' or 'regression'.")
    
    # If we made it here, the scoring metric is valid
    return True


def get_scoring_and_label(model_type, scoring_metric=None):
    """
    Returns the appropriate scoring metric and score label based on the model type and provided scoring metric.

    Args:
        model_type: 'classification' or 'regression' to specify the type of model.
        scoring_metric: Metric to evaluate the models ('R2', 'MAE', 'RMSE' for regression or 'accuracy', 'precision' for classification).
                       If not provided, defaults to 'accuracy' for classification and 'R2' for regression.
    
    Returns:
        scoring: The appropriate scoring metric to use in GridSearchCV.
        score_label: A formatted label string for displaying the best score.
    """

    # Set default scoring if not provided
    if scoring_metric is None:
        scoring_metric = 'accuracy' if model_type == 'classification' else 'r2'


    # Validate the scoring metric
    is_valid_scoring_metric(model_type, scoring_metric)

    # Select the appropriate scoring and score label using the SCORING_MAP
    scoring = SCORING_MAP[model_type][scoring_metric.lower()]
    score_label = f'Best {scoring_metric.capitalize()}' if model_type == 'classification' else f'Best {scoring_metric.upper()}'
    
    return scoring, score_label



def train_baseline_model(X_train, y_train, model_type='classification', scoring=None, cv=5, training_time=True):
    """
    Trains a baseline model (DummyClassifier or DummyRegressor) based on the model type and returns the model, score, and training time.
    The baseline model uses default parameters.

    Args:
        X_train: Training features.
        y_train: Training target.
        model_type: 'classification' or 'regression' to specify the type of baseline model (default: 'classification').
        scoring_metric: Metric to evaluate the model ('R2', 'MAE', 'RMSE' for regression or 'accuracy', 'precision' for classification).
                       If not provided, defaults to 'accuracy' for classification and 'R2' for regression.
        cv: Number of cross-validation folds (default: 5).

    Returns:
        A dictionary containing the baseline model, score, and training time.
    """
    
    # Create baseline estimator based on model type
    if model_type == 'classification':
        baseline_model = DummyClassifier()
        model_name = 'Dummy Classifier'
    elif model_type == 'regression':
        baseline_model = DummyRegressor()
        model_name = 'Dummy Regressor'
    else:
        raise ValueError("Invalid model type. Choose 'classification' or 'regression'.")

    print(f"Training baseline {model_name} model...")

    # Get scoring and score label using the helper function
    scoring, score_label = get_scoring_and_label(model_type, scoring)

    # Record start time
    selection_start_time = time.time()

    # Initialize GridSearchCV with the baseline model and default parameters
    grid_search = GridSearchCV(estimator=baseline_model, param_grid={}, scoring=scoring, cv=cv)
    grid_search.fit(X_train, y_train)

    # Record end time
    selection_end_time = time.time()

    # Calculate the time it took to train the model
    selection_elapsed_time = selection_end_time - selection_start_time

    # Get the best model and score (with default parameters)
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_

    training_elapsed_time = -1

    if training_time:
        training_start_time = time.time()
        best_model.fit(X_train, y_train)
        training_end_time = time.time()
        training_elapsed_time = training_end_time - training_start_time

    # Return the results including the training time
    return {
        'Model Name': model_name,
        'Best Parameters': {},
        'Model': best_model,
        score_label: best_score,
        'Score': best_score,
        'Training Time (seconds)': training_elapsed_time,
        'Selection Time (seconds)': selection_elapsed_time
    }

def train_models( 
        models_list, 
        pipeline, 
        X_train, 
        y_train, 
        model_type='classification', 
        scoring =None, 
        cv=5, 
        training_time=True, 
        verbose=0):
    """
    Trains multiple models using GridSearchCV and returns the best model, parameters, score, and training 
    time for each model based on the scoring metric. trianing time is optional as it requires an extra refit step.
    """

    results = []

        # Get scoring and score label
    scoring_metric, score_label = get_scoring_and_label(model_type, scoring)
    
    for model_obj in models_list:
        # Access ModelWrapper attributes using dot notation
        estimator = model_obj.estimator
        param_grid = model_obj.param_grid
        model_name = model_obj.name
        
        if verbose > 0:
            print(f"Training {model_name} model...")

        # Create a deep copy of the pipeline and set the model estimator
        pipeline_copy = copy.deepcopy(pipeline)
        pipeline_copy.set_params(model=estimator)

        # Record start time
        selection_start_time = time.time()

        # Initialize GridSearchCV with the appropriate scoring metric and cross-validation
        grid_search = GridSearchCV(estimator=pipeline_copy, param_grid=param_grid, scoring=scoring_metric, cv=cv, verbose=verbose, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Record end time
        selection_end_time = time.time()

        # Calculate the time it took to select the model
        selection_elapsed_time = selection_end_time - selection_start_time

        # Get the best model, parameters, and score
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        training_elapsed_time = -1

        # Calculate the time it took to train the model
        if training_time:
            training_start_time = time.time()
            best_model.fit(X_train, y_train)
            training_end_time = time.time()
            training_elapsed_time = training_end_time - training_start_time

        # Append results, including the training time, best parameters, and best score
        results.append({
            'Model Name': model_name,
            'Best Parameters': best_params,
            'Model': best_model,
            score_label: best_score,
            'Score': best_score,
            'Training Time (seconds)': training_elapsed_time,
            'Selection Time (seconds)': selection_elapsed_time
        })
    
    return results


def plot_roc_curve(models_list, X_test, y_test):
    """
    Function to plot ROC curves for multiple models provided in a list of dictionaries.

    :param models_list: List of dictionaries with each containing:
                        - 'Model Name': Name of the model
                        - 'Best Parameters': Best hyperparameters for the model
                        - 'Model': Trained model object
                        
    :param X_test: Test feature set
    :param y_test: True labels for the test set
    """
    plt.figure(figsize=(10, 8))

    for model_info in models_list:
        model_name = model_info['Model Name']
        model = model_info['Model']

        # Predict probabilities
        y_probs = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for this model
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")

    # Plot diagonal line for random guessing
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison for All Models')
    plt.legend(loc="lower right")
    plt.show()


def plot_feature_importance(model, X_train, X_test, y_test, n_repeats=10, random_state=42):
    """
    Function to calculate and plot the feature importance for a given model.
    
    Parameters:
    - model: The trained model to evaluate.
    - X_train: Training data used to extract feature names.
    - X_test: Test data to calculate permutation importance.
    - y_test: True labels for the test data.
    - n_repeats: Number of times to shuffle a feature for importance calculation (default 10).
    - random_state: Random seed for reproducibility (default 42).
    """
    estimator = model['Model']
    # Calculate permutation importance
    result = permutation_importance(estimator, X_test, y_test, n_repeats=n_repeats, random_state=random_state)
    
    # Create a dataframe to display feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,  # Assumes X_train is a DataFrame
        'Importance': result.importances_mean
    })
    
    # Sort by importance and display the result
    sorted_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(sorted_df)
    
    model_name = model['Model Name']
    # Optional: Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_df['Feature'], sorted_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title(f'Feature Importance for {model_name} Model')
    plt.gca().invert_yaxis()  # Invert the y-axis to have the most important at the top
    plt.show()

def create_performance_bar_chart(ax, data, metric_name, models):
    colors = cm.get_cmap('viridis', len(models))  # Generate colors from colormap
    ax.bar(models, data, color=colors(range(len(models)))) 
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)
    ax.set_xticks(range(len(models)), models, rotation=45, ha='right')

# Function to plot model performance based on model type from a DataFrame
def plot_model_performance(results_df, model_type, graph_title=None):
    """
    Function to plot performance metrics for either classification or regression models.
    
    Args:
    results_df (pd.DataFrame): A DataFrame containing model names and their performance metrics.
    model_type (str): Specify 'classification' or 'regression' to adjust the metrics plotted.
    
    """
    models = results_df['Model Name']  # Extract model names

    # Set up figure with subplots for each metric
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    if graph_title:
        fig.suptitle(graph_title)
    fig.tight_layout()

    if model_type == 'classification':
        # Extract classification metrics from the DataFrame
        accuracy_values = results_df['Accuracy']
        precision_values = results_df['Precision']
        recall_values = results_df['Recall']
        f1_values = results_df['F1']

        create_performance_bar_chart(axs[0, 0], accuracy_values, 'Accuracy', models)
        create_performance_bar_chart(axs[0, 1], precision_values, 'Precision', models)
        create_performance_bar_chart(axs[1, 0], recall_values, 'Recall', models)
        create_performance_bar_chart(axs[1, 1], f1_values, 'F1', models)  
    else:
        # Extract regression metrics from the DataFrame
        rmse_values = results_df['RMSE']
        r2_values = results_df['R2']
        mae_values = results_df['MAE']
        mse_values = results_df['MSE']

        create_performance_bar_chart(axs[0, 0], rmse_values, 'RMSE', models)
        create_performance_bar_chart(axs[0, 1], r2_values, 'R2', models)
        create_performance_bar_chart(axs[1, 0], mae_values, 'MAE', models)
        create_performance_bar_chart(axs[1, 1], mse_values, 'MSE', models)
    # Adjust layout
    plt.tight_layout()
    if graph_title:
        plt.savefig(f'images/{graph_title}.png')
    plt.show()


def evaluate_models(best_models, X, y, model_type='classification', scoring=None, baseline_model=None, graph_title=None):
    """
    Evaluates the best models on the given feature set (X) and target values (y),
    and displays a table with model performance.

    Args:
        best_models (list): List of dictionaries containing best models and associated metadata.
        X (pd.DataFrame or np.array): Feature set (can be training or test).
        y (pd.Series or np.array): Target values (can be training or test).
        model_type (str): The type of model ('classification' or 'regression').
        scoring (str): The metric to score models on.

    Returns:
        None
    """
    # Get the appropriate scoring metric and score label
    # scoring, score_label = get_scoring_and_label(model_type, scoring)

    # Validate the scoring metric
    is_valid_scoring_metric(model_type, scoring)

    # Copy the list of best models to evaluate to avoid modifying the original list
    models_to_evaluate = best_models.copy()

    if baseline_model is not None:
        models_to_evaluate.insert(0, baseline_model)

    results_table = []
    
    for model_info in models_to_evaluate:
        model = model_info['Model']
        model_name = model_info['Model Name']
        best_params = model_info['Best Parameters']
        training_elapsed_time = model_info['Training Time (seconds)']
        selection_elapsed_time = model_info['Selection Time (seconds)']
        
        # Predictions on the feature set
        y_pred = model.predict(X)
        if model_type == 'classification':
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            results_table.append({
                'Model Name': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Best Parameters': best_params,
                'Training Time (seconds)': training_elapsed_time,
                'Selection Time (seconds)': selection_elapsed_time
            })
        else:
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mae = np.mean(np.abs(y - y_pred))
            mse = mean_squared_error(y, y_pred)
            results_table.append({
                'Model Name': model_name,
                'RMSE': rmse,
                'R2': r2,
                'MAE': mae,
                'MSE': mse,
                'Best Parameters': best_params,
                'Training Time (seconds)': training_elapsed_time,
                'Selection Time (seconds)': selection_elapsed_time
            })

    results_df = pd.DataFrame(results_table)
    

    # Sort and display the results based on the selected metric
    if scoring:
        if model_type == 'classification':
            results_df = results_df.sort_values(by=scoring.capitalize(), ascending=False)
        else:
            results_df = results_df.sort_values(by=scoring.upper(), ascending=True)
    
    # Mark the best model
    results_df.iloc[0, results_df.columns.get_loc('Model Name')] += ' *'
    
    # Display results and plots
    print(tabulate(results_df, headers='keys', tablefmt='grid'))

    plot_model_performance(results_df, model_type, graph_title)
    return results_df


def get_best_model(models_list, model_type, scoring):
    """
    Function to find and return the model with the best score based on the provided scoring metric.
    
    :param models_list: List of dictionaries where each dictionary contains:
                        - 'Model Name': Name of the model
                        - 'Best Parameters': Best hyperparameters for the model
                        - 'Model': Trained model object
                        - 'Score': Precomputed score of the model (cross-validated or test score)
    :param scoring: The scoring metric (e.g., 'mse', 'r2', 'accuracy', 'f1', 'precision', 'recall').
                    Determines how the score should be interpreted (maximize or minimize).
    :return: Dictionary of the best model and its score based on the specified scoring metric.
    """

    scoring = scoring.lower()

    # Validate the scoring metric
    is_valid_scoring_metric(model_type, scoring)

    best_model = None
    best_score = None

    # Define if we are maximizing or minimizing based on the scoring metric
    if scoring in ['mse', 'rmse', 'mae', 'msle']:  # Minimize for these metrics
        best_score = float('inf')
    elif scoring in ['r2', 'accuracy', 'f1', 'precision', 'recall', 'roc_auc']:  # Maximize for these metrics
        best_score = float('-inf')

    for model in models_list:
        model_score = model['Score']

        # Compare scores based on whether we are maximizing or minimizing
        if (scoring in ['mse', 'rmse', 'mae', 'msle'] and model_score < best_score) or \
           (scoring in ['r2', 'accuracy', 'f1', 'precision', 'recall', 'roc_auc'] and model_score > best_score):
            best_score = model_score
            best_model = model

    return best_model


def calculate_optimal_pca_components(X, preprocessor, threshold=0.95):
    """
    Calculates the optimal number of PCA components based on the explained variance threshold.

    Parameters:
    - X: The dataset (features) on which PCA will be applied.
    - preprocessor: The preprocessing pipeline (e.g., ColumnTransformer) to apply before PCA.
    - threshold: The cumulative explained variance threshold to select the number of components (default is 0.95).

    Returns:
    - optimal_components: The optimal number of PCA components that explain the given threshold of variance.
    - cumulative_explained_variance: The cumulative explained variance for each number of components.
    """
    
    # Create a pipeline that includes the preprocessor and PCA
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA())  # PCA without specifying n_components to calculate all
    ])
    
    # Fit the pipeline to the data
    pipeline.fit(X)
    
    # Extract the PCA step from the pipeline
    pca = pipeline.named_steps['pca']
    
    # Calculate cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot the cumulative explained variance
    plt.plot(cumulative_explained_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs Number of Components')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'{int(threshold * 100)}% variance')
    plt.axvline(x=np.argmax(cumulative_explained_variance >= threshold) + 1, color='g', linestyle='--', 
                label=f'Optimal Components: {np.argmax(cumulative_explained_variance >= threshold) + 1}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Find the optimal number of components that meet or exceed the variance threshold
    optimal_components = np.argmax(cumulative_explained_variance >= threshold) + 1
    
    print(f"Optimal number of components to explain {int(threshold * 100)}% of the variance: {optimal_components}")
    
    return optimal_components, cumulative_explained_variance

