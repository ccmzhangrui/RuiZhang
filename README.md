# https://github.com/ARDS-DynamicPhenotypes
The clustering algorithm employed soft dynamic time warping (soft-DTW) to account for temporal shifts in disease progression:

PYTHON
def temporal_clustering(X, n_clusters=3):
    enhanced_features = np.concatenate([
        X,
        np.gradient(X, axis=1),  # Rate of change
        savgol_filter(X, window=5, polyorder=3)  # Smoothed trends
    ], axis=2)
    
    clusterer = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="softdtw",
        metric_params={"gamma": 0.5},
        random_state=42
    )
    return clusterer.fit_predict(enhanced_features)
