import pandas as pd
import numpy as np
import h5py

import matplotlib.pyplot as plt

class tools:
    def histogram(feature, bins=100):
        feature = np.array(feature, dtype=np.int32)
        counts, bin = np.histogram(feature, bins)
        plt.stairs(counts, bin)
        plt.show()

    def catagorical_bar(feature):
        codes, uniques = pd.factorize(feature)
        x, y = np.unique(codes, return_counts=True)
        plt.bar(x, y, tick_label=uniques.values)
        plt.show()


class space:
    def __init__(self, file_dir='datasets/h5py_uniprot.h5'):
        self.file = h5py.File(file_dir, 'r')
        self.features = self.file.keys()

    def get_length_hist(self, bin=100):
        counts, bins = np.histogram(self.file['length'], bin)
        plt.stairs(counts, bins, fill=True)
        plt.show()

class raw_tsv:
    def __init__(self, file_path='datasets/uniprot_tokenised_db.tsv', maxsize=1000000, save=False) -> None:
        samples = []
        with open(file_path, 'r') as f:
            n = len(f.readlines())

        with open(file_path, 'r') as f:
            line = f.readline()
            columns = line.split('\t')
            while line:
                line = f.readline()
                row = line.split('\t')
                if len(row) == len(columns):
                    samples.append(row)
        samples = samples[:maxsize]
        self.df = pd.DataFrame(samples, columns=columns)
        if save:
            self.df.to_excel('datasets/datasets_df.xlsx')

    def protein_length_hist(self):
        tools.histogram(self.df['length'])

    def enyzme_class(self):
        tools.catagorical_bar(self.df['ecId'])

    def organsisms_distrib(self):
        tools.catagorical_bar(self.df['taxId'])

class premade:
    def __init__(self, df_path='datasets/datasets_df.xlsx') -> None:
        self.df = pd.read_excel(df_path)

    def protein_length_hist(self):
        tools.histogram(self.df['length'])

    def enyzme_class(self):
        tools.catagorical_bar(self.df['ecId'])

    def organsisms_distrib(self):
        tools.catagorical_bar(self.df['taxId'])

    
if __name__ == '__main__':
    x = raw_tsv()
    # x = premade()
    # x.protein_length_hist()
    x.enyzme_class()
    x.organsisms_distrib()