import pandas as pd
import hdbscan
import time
import sys

df_bitcoin = pd.read_csv('datasets/BitcoinHeistData.csv')
df_bitcoin = df_bitcoin.drop(['address', 'label'], axis = 1)
df_bitcoin.head(800000).to_csv('datasets/bitcoin.csv', sep=',', index = False)
print(df_bitcoin.head(800000))
sys.exit(1)

#
df_labels = pd.read_csv('labels_hdbscan.csv')
print(df_labels)
print("Clusters:", df_labels['0'].unique())
print("Clusters:", max(df_labels['0'].unique()))
print("Clusters:", df_labels['0'].nunique())
sys.exit(1)

df_sensor = pd.read_csv('datasets/sensor_stream_1m.csv')
print(df_sensor)

start = time.time()
clusterer = hdbscan.HDBSCAN(min_cluster_size = 10, min_samples = 5, match_reference_implementation = True, core_dist_n_jobs = 5)
clusterer.fit(df_sensor)
labels    = clusterer.labels_
end   = start = time.time()

df_labels = pd.DataFrame(labels)
df_labels.to_csv('labels_hdbscan.csv', index=False)

print("Clusters:", df_labels['0'].unique())
print("Clusters:", df_labels['0'].nunique())