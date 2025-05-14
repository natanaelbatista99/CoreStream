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
# parallelism
from multiprocessing import Pool, cpu_count

class Evaluation(nn.Module):

    def __init__(self, dataset = '', mpts = [], timestamp = 0):
        self.base_dir_result    = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "results/" + str(dataset) + "/")
        self.mpts               = sorted(mpts)
        self.timestamp          = timestamp
        self.partitions_bubbles = None
        self.partitions_hdbscan = None

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
        plt.xticks(fontsize=16)  # Eixo X
        plt.yticks(fontsize=16)  # Eixo Y
        plt.xlabel("HDBSCAN* Partition", fontsize=26)
        plt.ylabel("CoreStream Partition", fontsize=26)
        plt.savefig(sub_dir + f"heatmap_{mensure['metric']}.png")
        plt.close()

    # Função auxiliar para calcular ARI e NMI
    def compute_scores(self, args):
        i, j, mpts_i, mpts_j = args

        ari = adjusted_rand_score(self.partitions_bubbles[mpts_i], self.partitions_hdbscan[mpts_j])
        nmi = normalized_mutual_info_score(self.partitions_bubbles[mpts_i], self.partitions_hdbscan[mpts_j])

        return (i, j, mpts_i, ari, nmi)

    def evaluation_mensure(self):
        # ARI partitions
        print(self.timestamp)

        sub_dir = os.path.join(self.base_dir_result, f"evaluation/timestamp_{self.timestamp}/")

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        df_partitions_bubbles, df_partitions_hdbscan = self.flat_partitions(self.timestamp, 'bubbles'), self.flat_partitions(self.timestamp, 'hdbscan')

        len_mpts    = len(self.mpts)
        df_mensures = pd.DataFrame(0.0, index=[x for x in self.mpts], columns=[['ari', 'nmi']])
        heatmaps    = {
            'ari': np.zeros((len_mpts, len_mpts)),
            'nmi': np.zeros((len_mpts, len_mpts))
        }

        self.partitions_bubbles = {mpts: df_partitions_bubbles[df_partitions_bubbles['partition_mpts'] == mpts].drop('partition_mpts', axis=1).values.ravel() for mpts in self.mpts}
        self.partitions_hdbscan = {mpts: df_partitions_hdbscan[df_partitions_hdbscan['partition_mpts'] == mpts].drop('partition_mpts', axis=1).values.ravel() for mpts in self.mpts}

        # Lista de argumentos para paralelização
        args_list = [
            (i, j, self.mpts[i], self.mpts[j])
            for i in range(len_mpts)
            for j in range(len_mpts)
        ]

        # Executa em paralelo
        with Pool(cpu_count() - 10) as pool:
            results = pool.map(self.compute_scores, args_list)

        # Preenche os resultados
        for i, j, mpt_i, ari, nmi in results:
            heatmaps['ari'][i][j] = ari
            heatmaps['nmi'][i][j] = nmi
            if i == j:
                df_mensures.at[mpt_i, 'ari'] = ari
                df_mensures.at[mpt_i, 'nmi'] = nmi

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

        del df_partitions_bubbles
        del df_partitions_hdbscan