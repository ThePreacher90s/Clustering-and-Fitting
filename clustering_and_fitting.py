"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
from numpy.polynomial import Polynomial as Poly


def plot_relational_plot(df):
    """
    Plotting scatter plots to check the
    relationships between petal-lenght
    and the species of iris flower.
    """
    # Creates a figure (fig) and an axis (ax) for plotting.
    # dpi = 144 Increases the resolution to display better.
    fig, ax = plt.subplots(dpi=144)
    # plotting a scatter plot, the label was used to produce a good legend
    sns.scatterplot(data=df, x='petal_length', y='petal_width',
                    hue='species', ax=ax)
    # Change lgend location to any upper left corner and titlted the legend.
    ax.legend(title='Species', loc='upper left')
    # Dynamic axis labels
    ax.set_xlabel(df.columns[2])
    ax.set_ylabel(df.columns[3])
    ax.set_title("Petal Length vs Petal Width")
    # Save and show the plot
    plt.savefig('relational_plot.png')
    plt.show()
    return


def plot_categorical_plot(df):
    """
    Plots a pie chart showing the
    distribution of different Iris species.
    """
    # Use group-by to count each species
    species_counts = df.groupby('species').size()
    # Create figure and axis
    fig, ax = plt.subplots(dpi=144)
    # Plot Pie Chart using `ax`
    species_counts.plot(
        ax=ax, kind='pie', autopct='%1.1f%%', startangle=200,
        colors=['lightblue', 'lightgreen', 'lightcoral']
    )
    # Set title and remove y-label for clarity
    ax.set_title("Distribution of Iris Species")
    ax.set_ylabel('')
    # Ensure the pie chart is a perfect circle
    ax.axis('equal')
    # Save the plot before displaying
    plt.savefig('categorical_plot.png')
    # Show the plot
    plt.show()
    return


def plot_statistical_plot(df):
    """
    Plots a correlation heatmap showing the pairwise
    relationships between numeric features in the dataset.
    Only the lower triangle of the matrix is shown to
    avoid redundant information.
    """
    # Create the figure and axis object
    fig, ax = plt.subplots(dpi=150)
    # Compute the correlation matrix for numeric columns
    corr = df.corr(numeric_only=True)
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))
    # Plot the heatmap
    sns.heatmap(
        corr, ax=ax, mask=mask, annot=True, cmap='RdBu',
        vmin=-1, vmax=1
    )
    # Add title to the heatmap
    ax.set_title('Statistical Correlation Heatmap',
                 fontsize=12, pad=12)
    # Adjust layout
    plt.tight_layout()
    # Save the plot
    plt.savefig('statistical_plot.png')
    # Display the heatmap
    plt.show()
    return


def statistical_analysis(df, col: str):
    """
    Computes statistical properties
    (mean, standard deviation, skewness, and kurtosis)
    for a given numerical column in the DataFrame.
    """
    # Compute statistics
    mean = df[col].mean()  # Mean (average)
    stddev = df[col].std()  # Standard Deviation (spread of values)
    skew = ss.skew(df[col], nan_policy='omit')  # Skewness (asymmetry)
    excess_kurtosis = ss.kurtosis(df[col], nan_policy='omit')  # ExcessKurtosis
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Cleans the dataset by removing duplicates,
    handling missing values, and printing key
    insights such as summary statistics, first
    few rows, and correlation matrix.
    """
    # Drop duplicate rows if any
    df = df.drop_duplicates()
    # Drop rows with missing values if any
    df = df.dropna()
    # make use of quick features such as 'describe', 'head/tail' and 'corr'
    # Display basic summary statistics
    print("Basic summary of the data:\n", df.describe())
    # Displays the first five rows of the data by default
    print("\nHead(the first five rows):\n", df.head())
    # Computes correlation matrix for numerical columns
    print("\nThe correlations of numerical data:\n",
          df.corr(numeric_only=True))
    # Returns the preprocessed data
    return df


def writing(moments, col):
    """
    Prints statistical moments and interprets the skewness and kurtosis
    of a given attribute.

    Args:
        moments (list or array-like): A sequence of four statistical values
        in the order [mean, standard deviation, skewness, excess kurtosis].
        col (str): The name of the attribute being analyzed.

    Prints:
        - Mean and standard deviation of the attribute.
        - Skewness and excess kurtosis with interpretation:
            - Skewness: left-skewed, right-skewed, or not skewed.
            - Kurtosis: platykurtic, mesokurtic, or leptokurtic.
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    # Delete the following options as appropriate for your data.
    # Not skewed and mesokurtic can be defined with asymmetries <-2 or >2.
    # print('The data was right/left/not skewed and platy/meso/leptokurtic.')
    # Interpret skewness based on asymmetry definition <-2 or >2
    if moments[2] < -2:
        skewness_type = "left-skewed"
    elif moments[2] > 2:
        skewness_type = "right-skewed"
    else:
        skewness_type = "not skewed"
    # Interpret kurtosis based on defined criteria
    if moments[3] < -1:
        kurtosis_type = "platykurtic"
    elif moments[3] > 1:
        kurtosis_type = "leptokurtic"
    else:
        kurtosis_type = "mesokurtic"
    print(f'The data was {skewness_type} and {kurtosis_type}.')
    return


def perform_clustering(df, col1, col2):
    """
    Performs KMeans clustering on two selected columns of a DataFrame,
    using silhouette score to choose the optimal number of clusters,
    and plots the Elbow Method for visual evaluation.
    Args:
        df (DataFrame): Input DataFrame.
        col1 (str): Name of the first column for clustering.
        col2 (str): Name of the second column for clustering.
    Returns:
       tuple: Cluster labels, original data, x/y cluster centers,
               and predicted labels for cluster centers.
    """

    def plot_elbow_method():
        """
        Plots the Elbow Method graph to help determine the optimal
        number of clusters (k) in a clustering algorithm.

        The plot displays the Within-Cluster Sum of Squares (WCSS)
        for a range of k values, and highlights the best k value
        using a vertical dashed line.
        """
        ks = list(range(2, 11))
        # Create the figure and axis object
        fig, ax = plt.subplots(dpi=144, figsize=(8, 5))
        # Plot WCSS values
        ax.plot(ks, wcss, marker='o')
        # Highlight the best k with a vertical line
        ax.axvline(x=best_n, color='r', linestyle='--',
                   label=f'Best k = {best_n}')
        # Set titles and labels
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('WCSS (Inertia)')
        # Add legend and grid
        ax.legend()
        ax.grid(True)
        # Save the plot
        plt.savefig('elbow_plot.png')
        # Show the plot
        fig.tight_layout()
        fig.show()
        return

    def one_silhouette_inertia():
        """
        Computes the silhouette score and inertia for KMeans clustering.
        Returns:
            tuple: A tuple containing:
                - silhouette_score (float): A measure of how similar an object
                is to its own cluster compared to other clusters.
                - inertia (float): Sum of squared distances of samples to
                their closest cluster center (WCSS).
        """
        kmeans = KMeans(n_clusters=n, n_init=20)
        kmeans.fit(norm)
        _score = silhouette_score(norm, kmeans.labels_)
        _inertia = kmeans.inertia_
        return _score, _inertia

    # === Gather data and scale ===
    df_cut = df[[col1, col2]].copy()
    scaler = StandardScaler()
    norm = scaler.fit_transform(df_cut)
    data = scaler.inverse_transform(norm)
    # === Find best number of clusters ===
    wcss = []
    best_n, best_score = None, -np.inf
    for n in range(2, 11):
        # Iterate over k from 2 to 10, compute silhouette score and inertia.
        # for each clustering result, and keep track of the best k value.
        _score, _inertia = one_silhouette_inertia()
        wcss.append(_inertia)
        print(f"{n} clusters silhouette score = {_score:.2f}")
        if _score > best_score:
            best_n = n
            best_score = _score

    # Prints the best silhouette score
    print(f"\nBest number of clusters = {best_n} "
          f"with silhouette score = {best_score:.2f}")
    # plot elbow method
    plot_elbow_method()

    # === Fit KMeans with best k and get cluster centers ===
    kmeans = KMeans(n_clusters=best_n, n_init=20, random_state=42)
    kmeans.fit(norm)
    labels = kmeans.labels_
    cen = scaler.inverse_transform(kmeans.cluster_centers_)
    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]
    cenlabels = kmeans.predict(kmeans.cluster_centers_)
    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """
    Plots clustered data points along with their corresponding cluster centers.

    This function visualizes the results of KMeans clustering by plotting
    the data points colored by their assigned cluster and overlaying the
    cluster centers with distinct markers.

    Args:
        labels (array-like): Cluster labels assigned to each data point.
        data (array-like): Original input data (2D array) used for clustering.
        xkmeans (array-like): X-coordinates of the cluster centers.
        ykmeans (array-like): Y-coordinates of the cluster centers.
        centre_labels (array-like): Cluster labels assigned to cluster centers
    """
    fig, ax = plt.subplots(dpi=144)
    colours = plt.cm.Set1(np.linspace(0, 1, len(np.unique(labels))))
    cmap = ListedColormap(colours)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap,
                         marker='o', label='Data')
    ax.scatter(xkmeans, ykmeans, c=centre_labels, cmap=cmap, marker='x',
               s=100, label='Centres')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_ticks(np.unique(labels))
    ax.legend()
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Sepal Width')
    plt.grid(True)
    plt.savefig('clustering.png')
    # Show the plot
    plt.show()
    return


def perform_fitting(df, col1, col2):
    """
    Fit a linear model to two columns of a DataFrame using numpy.

    Parameters:
    df (DataFrame): Input pandas DataFrame containing the data.
    col1 (str): Column name to use as the independent variable (x).
    col2 (str): Column name to use as the dependent variable (y).

    Returns:
    tuple: A dictionary with fitted model details, and the original x and y.
    """

    # Gather data and prepare for fitting
    x = df[col1].values
    y = df[col2].values
    # Fit model using numpy Polynomial
    p = Poly.fit(x, y, 1)  # degree 1 linear fit
    cov = np.polyfit(x, y, 1, cov=True)[1]  # Covariance matrix
    sigma = np.sqrt(np.diag(cov))
    # Extract standard coefficients (convert from shifted basis)
    b, a = p.convert().coef
    # Print coefficients and their uncertainties
    print(f"a = {a:.2f} +/- {sigma[0]:.2f}")
    print(f"b = {b:.2f} +/- {sigma[1]:.2f}")
    # Predict across x
    xfit = np.linspace(min(x), max(x), 100)
    yfit = a * xfit + b

    # Prepare data dictionary
    data = {
        'params': [a, b],
        'sigma': sigma,
        'xfit': xfit,
        'yfit': yfit
    }
    return data, x, y


def plot_fitted_data(data, x, y):
    """
    Plots the original data along with a fitted line and an error band.
    Parameters:
    data (dict): A dictionary containing:
        - 'xfit' (array-like): X values for the fitted line.
        - 'yfit' (array-like): Y values for the fitted line.
        - 'params' (tuple): Parameters (a, b) of the fitted linear model.
        - 'sigma' (tuple): Standard deviations (sigma_a, sigma_b) for a and b.
    x (array-like): The x-values of the original data.
    y (array-like): The y-values of the original data.
    """
    def linfunc(x, a, b):
        """Linear function a * x + b."""
        return a * x + b

    fig, ax = plt.subplots(dpi=144)
    # Original scatter plot from data
    ax.plot(x, y, 'bo', label='Data')
    # Fitted line
    xfit = data['xfit']
    yfit = data['yfit']
    ax.plot(xfit, yfit, 'k-', label='Fit')
    # Error band
    a, b = data['params']
    sigma_a, sigma_b = data['sigma']
    ax.fill_between(
        xfit,
        linfunc(xfit, a - sigma_a, b - sigma_b),
        linfunc(xfit, a + sigma_a, b + sigma_b),
        color='k',
        alpha=0.1,
        label='Fit Error'
    )
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.legend()
    plt.savefig('fitting.png')
    plt.show()
    return


def main():
    """
    Run the full data analysis pipeline including preprocessing, plotting,
    statistical analysis, clustering, and curve fitting.
    """
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'petal_length'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    clustering_results = perform_clustering(df, 'petal_length', 'sepal_width')
    plot_clustered_data(*clustering_results)
    fitting_results = perform_fitting(df, 'petal_length', 'petal_width')
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()
