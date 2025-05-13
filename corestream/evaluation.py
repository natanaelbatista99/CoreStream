import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import duckdb
import os
import re
import sys
import torch.nn as nn
import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

class Evaluation(nn.Module):

    def __init__(self, dataset = '', mpts = [], timestamp = 0):
        self.base_dir_result = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "results/" + str(dataset) + "/")
        self.mpts            = sorted(mpts)
        self.timestamp       = timestamp

    def flat_partitions(self, timestamp, partition):
        # RETURN DATAFRAME BUBBLES/HDBSCAN PARTITIONS
        query         = f"SELECT * FROM '{self.base_dir_result}flat_solutions/flat_solution_partitions_t{timestamp}/partitions_objects/partition_{partition}_mpts_*.parquet'"
        df_partitions = duckdb.query(query).to_df()
        df_partitions = df_partitions.drop('__index_level_0__', axis=1)

        return df_partitions

    def plot_heatmap(self, sub_dir, objects, df_heatmap, mensure):
        sns.set(rc={"figure.figsize":(16, 14)})
        ax = sns.heatmap(df_heatmap)

        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=26)  # Tamanho dos números da barra
        colorbar.ax.yaxis.label.set_size(26)   # Tamanho do texto da barra (se houver)
        colorbar.set_label(mensure['label'], fontsize=30)

        plt.title("Heatmap TIMESTAMP " + str(self.timestamp) + " | # Objects: " + str(objects) + " | Mean: " + str(round(df_heatmap.values.mean(), 4)) + " | std: " + str(round(df_heatmap.values.std(), 4)), fontsize=26)
        plt.xticks(fontsize=20)  # Eixo X
        plt.yticks(fontsize=20)  # Eixo Y
        plt.xlabel("HDBSCAN* Partition", fontsize=26)
        plt.ylabel("CoreStream Partition", fontsize=26)
        plt.savefig(sub_dir + f"heatmap_{mensure['metric']}.png")
        plt.close()

    def evaluation_mensure(self):
        # ARI partitions
        print(self.timestamp)

        sub_dir = os.path.join(self.base_dir_result, f"evaluation/timestamp_{self.timestamp}/")

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        df_partitions_bubbles, df_partitions_hdbscan = self.flat_partitions(self.timestamp, 'bubbles'), self.flat_partitions(self.timestamp, 'hdbscan')

        len_mpts    = len(self.mpts)
        df_mensures = pd.DataFrame(0.0, index=[x for x in self.mpts], columns=[['ari', 'nmi']])
        heatmaps = {
            'ari': np.zeros((len_mpts, len_mpts)),
            'nmi': np.zeros((len_mpts, len_mpts))
        }

        for i in range(len_mpts):
            for j in range(len_mpts):
                partition_b = df_partitions_bubbles[df_partitions_bubbles['partition_mpts'] == self.mpts[i]].drop('partition_mpts', axis=1)
                partition_h = df_partitions_hdbscan[df_partitions_hdbscan['partition_mpts'] == self.mpts[j]].drop('partition_mpts', axis=1)

                heatmaps['ari'][i][j] = adjusted_rand_score(partition_b.values.ravel(), partition_h.values.ravel())
                heatmaps['nmi'][i][j] = normalized_mutual_info_score(partition_b.values.ravel(), partition_h.values.ravel())

                if i == j:
                    df_mensures.at[self.mpts[i], 'ari'] = heatmaps['ari'][i][j]
                    df_mensures.at[self.mpts[i], 'nmi'] = heatmaps['nmi'][i][j]

        df_heatmap_ari = pd.DataFrame(heatmaps['ari'], index=self.mpts, columns=self.mpts)
        df_heatmap_ari.to_csv(sub_dir + "heatmap_ari.csv")

        df_heatmap_nmi = pd.DataFrame(heatmaps['nmi'], index=self.mpts, columns=self.mpts)
        df_heatmap_nmi.to_csv(sub_dir + "heatmap_nmi.csv")

        df_mensures.to_csv(sub_dir + "partitions_ari.csv")

        # HEATMAP PLOT
        mensure = {'metric': 'ari', 'label': 'Adjusted Rand Index'}
        self.plot_heatmap(sub_dir, df_partitions_bubbles.shape[1] - 1, df_heatmap_ari, mensure)

        mensure = {'metric': 'nmi', 'label': 'Normalized Mutual Informatio'}
        self.plot_heatmap(sub_dir, df_partitions_bubbles.shape[1] - 1, df_heatmap_nmi, mensure)