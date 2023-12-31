def hopkins(X):
    from math import isnan
    from numpy.random import uniform
    from sklearn.neighbors import NearestNeighbors
    from random import sample
    import numpy as np
    
    d = X.shape[1]
    n = len(X)
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X, axis=0), d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H

### ----------------------------------------###--------------------------------------###------------------------------------------###--------------------------------------------

def elbow(X):
    from yellowbrick.cluster import KElbowVisualizer
    from sklearn.cluster import KMeans
    from seaborn import set_palette
    import matplotlib.pyplot as plt
    
    # Figures Settings
    color_palette = ['#FFCC00', '#54318C']
    set_palette(color_palette)
    title = dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
    text_style = dict(fontweight='bold', fontfamily='serif')
    
    # Creating subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow Score
    elbow_score = KElbowVisualizer(KMeans(random_state=0, max_iter=500), k=(2, 10), ax=ax1, timings=False)
    elbow_score.fit(X)
    elbow_score.finalize()
    ax1.set_title('Distortion Score Elbow\n', **title)
    ax1.tick_params(labelsize=7)
    for text in ax1.legend_.texts:
        text.set_fontsize(9)
    for spine in ax1.spines.values():
        spine.set_color('None')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), borderpad=2, frameon=False, fontsize=8)
    ax1.grid(axis='y', alpha=0.5, color="black", linestyle='dotted')
    ax1.grid(axis='x', alpha=0)
    ax1.set_xlabel('\nK Values', fontsize=9, **text_style)
    ax1.set_ylabel('Distortion Scores\n', fontsize=9, **text_style)
    
    # Elbow Score (Calinski-Harabasz Index)
    elbow_score_ch = KElbowVisualizer(KMeans(random_state=0, max_iter=500), k=(2, 10), metric='calinski_harabasz', timings=False, ax=ax2)
    elbow_score_ch.fit(X)
    elbow_score_ch.finalize()
    ax2.set_title('Calinski-Harabasz Score Elbow\n', **title)
    ax2.tick_params(labelsize=7)
    for text in ax2.legend_.texts:
        text.set_fontsize(9)
    for spine in ax2.spines.values():
        spine.set_color('None')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), borderpad=2, frameon=False, fontsize=8)
    ax2.grid(axis='y', alpha=0.5, color="black", linestyle='dotted')
    ax2.grid(axis='x', alpha=0)
    ax2.set_xlabel('\nK Values', fontsize=9, **text_style)
    ax2.set_ylabel('Calinski-Harabasz Score\n', fontsize=9, **text_style)
    
    # Adding title and credits
    plt.suptitle('Credit Card Customer Clustering using K-Means', fontsize=14, **text_style)
    plt.gcf().text(0.9, 0.05, 'kaggle.com/caesarmario', style='italic', fontsize=7)
    
    # Adjusting layout and displaying the plots
    plt.tight_layout()
    plt.show()

### ----------------------------------------###--------------------------------------###------------------------------------------###--------------------------------------------

def visualizerScatterwithout(X, kmeans, y_kmeans, num_clusters=None, colors_c=None):
    from pywaffle import Waffle
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Define your figure settings
    cluster_colors = colors_c
    labels = [f'Cluster {i + 1}' for i in range(num_clusters)]  # Cluster labels
    title = dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
    text_style = dict(fontweight='bold', fontfamily='serif')
    scatter_style = dict(linewidth=0.65, edgecolor='#100C07', alpha=0.85)
    legend_style = dict(borderpad=2, frameon=False, fontsize=8)

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'hspace': 0.5})

    # --- Clusters Distribution ---

    # Subplot 1 (ax1): Scatter plot of cluster distributions
    y_kmeans_labels = list(set(y_kmeans.tolist()))
    for i in y_kmeans_labels:
        ax1.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=50, c=cluster_colors[i], **scatter_style)
    ax1.set_title('Scatter Plot Clusters Distributions\n', **title)
    ax1.legend(labels, bbox_to_anchor=(0.95, -0.05), ncol=num_clusters, **legend_style)
    ax1.grid(axis='both', alpha=0.5, color='#9B9A9C', linestyle='dotted')
    ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    for spine in ax1.spines.values():
        spine.set_color('None')
    ax1.xaxis.grid(False)
    ax1.yaxis.grid(False)

    # Subplot 2 (ax2): Waffle chart (percentage of each cluster)
    unique, counts = np.unique(y_kmeans, return_counts=True)
    df_waffle = dict(zip(unique, counts))
    total = sum(df_waffle.values())

    # Calculate the total number of desired icons (100)
    total_icons_desired = 100

    # Calculate the scale factor to adjust the size of the icons
    scale_factor = total_icons_desired / total

    # Adjust the number of icons for each cluster
    wfl_square = {key: round(value * scale_factor) for key, value in df_waffle.items()}

    # Calculate the adjusted total icons
    total_icons_adjusted = sum(wfl_square.values())

    # Calculate the difference between adjusted and desired total
    difference = total_icons_desired - total_icons_adjusted

    # If there is a positive difference, add the difference to a random cluster
    if difference > 0:
        clusters = list(wfl_square.keys())
        while difference > 0:
            cluster = np.random.choice(clusters)
            wfl_square[cluster] += 1
            difference -= 1

    # If there is a negative difference, subtract the difference from a random cluster
    elif difference < 0:
        clusters = list(wfl_square.keys())
        while difference < 0:
            cluster = np.random.choice(clusters)
            if wfl_square[cluster] > 0:
                wfl_square[cluster] -= 1
                difference += 1

    # Calculate the adjusted percentages
    wfl_label = {key: round(value / total_icons_desired * 100, 2) for key, value in wfl_square.items()}

    ax2.set_title('Percentage of Each Clusters\n', **title)
    ax2.set_aspect(aspect='auto')
    Waffle.make_waffle(ax=ax2, rows=6, values=wfl_square, colors=cluster_colors, 
                    labels=[f"Cluster {i+1} - ({k}%)" for i, k in wfl_label.items()], icons='child', icon_size=30, 
                    legend={'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.05), 'ncol': num_clusters, 'borderpad': 2, 
                            'frameon': False, 'fontsize': 10})

    # Display the figure
    plt.show()


def visualizerScatter(X, kmeans, y_kmeans, churn_labels, num_clusters=None, colors_c=None):
    from pywaffle import Waffle
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Define your figure settings
    cluster_colors = colors_c
    labels = [f'Cluster {i + 1}' for i in range(num_clusters)]  # Cluster labels
    title = dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
    text_style = dict(fontweight='bold', fontfamily='serif')
    scatter_style = dict(linewidth=0.65, edgecolor='#100C07', alpha=0.85)
    legend_style = dict(borderpad=2, frameon=False, fontsize=8)

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'hspace': 0.5})

    # --- Clusters Distribution ---

    # Subplot 1 (ax1): Scatter plot of cluster distributions
    y_kmeans_labels = list(set(y_kmeans.tolist()))
    for i in y_kmeans_labels:
        ax1.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=50, c=cluster_colors[i], **scatter_style)

    # Add black points for churn = 1 in the original dataset
    churn_points = X[churn_labels == 1]
    ax1.scatter(churn_points[:, 0], churn_points[:, 1], s=50, c='black', label='Churn = 1', alpha=0.7)

    ax1.set_title('Scatter Plot Clusters Distributions\n', **title)
    ax1.legend(labels + ['Churn = 1'], bbox_to_anchor=(0.95, -0.05), ncol=num_clusters + 1, **legend_style)
    ax1.grid(axis='both', alpha=0.5, color='#9B9A9C', linestyle='dotted')
    ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    for spine in ax1.spines.values():
        spine.set_color('None')
    ax1.xaxis.grid(False)
    ax1.yaxis.grid(False)

    # Subplot 2 (ax2): Waffle chart (percentage of each cluster)
    unique, counts = np.unique(y_kmeans, return_counts=True)
    df_waffle = dict(zip(unique, counts))
    total = sum(df_waffle.values())

    # Calculate the total number of desired icons (100)
    total_icons_desired = 100

    # Calculate the scale factor to adjust the size of the icons
    scale_factor = total_icons_desired / total

    # Adjust the number of icons for each cluster
    wfl_square = {key: round(value * scale_factor) for key, value in df_waffle.items()}

    # Calculate the adjusted total icons
    total_icons_adjusted = sum(wfl_square.values())

    # Calculate the difference between adjusted and desired total
    difference = total_icons_desired - total_icons_adjusted

    # If there is a positive difference, add the difference to a random cluster
    if difference > 0:
        clusters = list(wfl_square.keys())
        while difference > 0:
            cluster = np.random.choice(clusters)
            wfl_square[cluster] += 1
            difference -= 1

    # If there is a negative difference, subtract the difference from a random cluster
    elif difference < 0:
        clusters = list(wfl_square.keys())
        while difference < 0:
            cluster = np.random.choice(clusters)
            if wfl_square[cluster] > 0:
                wfl_square[cluster] -= 1
                difference += 1

    # Calculate the adjusted percentages
    wfl_label = {key: round(value / total_icons_desired * 100, 2) for key, value in wfl_square.items()}

    ax2.set_title('Percentage of Each Clusters\n', **title)
    ax2.set_aspect(aspect='auto')
    Waffle.make_waffle(ax=ax2, rows=6, values=wfl_square, colors=cluster_colors, 
                    labels=[f"Cluster {i+1} - ({k}%)" for i, k in wfl_label.items()], icons='child', icon_size=30, 
                    legend={'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.05), 'ncol': num_clusters, 'borderpad': 2, 
                            'frameon': False, 'fontsize': 10})

    # Display the figure
    plt.show()


    ### ----------------------------------------###--------------------------------------###------------------------------------------###--------------------------------------------

