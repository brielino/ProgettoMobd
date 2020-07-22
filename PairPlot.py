import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def pairP(datasetName):
    # metodo per calcolare il Plot di un dataset
    # bisogna passare il percorso del dataset
    sns.set(style="ticks", color_codes=True)
    dataset = pd.read_csv(datasetName)
    dataset.describe(include='all')
    sns_plot = sns.pairplot(dataset, hue='CLASS', height=2.5)
    plt.show()

