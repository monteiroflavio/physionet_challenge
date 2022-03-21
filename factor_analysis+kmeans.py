import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.cm import rainbow
import matplotlib.ticker as ticker
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import FactorAnalysis, PCA

TICK_SPACE = 1
TICK_FONT_SIZE = 15
MARK_SIZE = 70
AXIS_LABEL_SIZE = 15
data = pd.read_csv('./tested_datasets/set-a_4sigmas_complete_fa2_oblimin_kmeans.csv', delimiter=',', header=0)

def plot_decomposition(data
                          , plot_2d=False
                          , plot_kmeans=False
                          , selected_groups=None
                          , axis=(1,2,3)):
    if plot_2d:

        # define style
        plt.style.use('seaborn-whitegrid')
        fig, ax = plt.subplots()

        # lock max and min values
        ax.set_xlim((-5,5))
        ax.set_ylim((-5,5))

        # add x and y labels
        plt.xlabel('1st Principal Component', fontsize=AXIS_LABEL_SIZE)
        plt.ylabel('2nd Principal Component', fontsize=AXIS_LABEL_SIZE)

        # configure tick font size
        plt.xticks(fontsize=TICK_FONT_SIZE)
        plt.yticks(fontsize=TICK_FONT_SIZE)

        # configure tick spacing
        ax.xaxis.set_major_locator(ticker.MultipleLocator(TICK_SPACE))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(TICK_SPACE))
        

    else:
        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        plt.cla()
    
    label_column = 'kmeans_labels' if plot_kmeans else 'outcome'
    colors = ['palegreen', 'red']
    markers = ['o', '^']
    labels = ['alive', 'dead']
    if plot_kmeans:
        # TODO: adapt to diferent subgroup sizes
        markers = ['o', 'o', '^', '^']
        colors = ['palegreen', 'dodgerblue', 'red', 'navajowhite', 'k', 'w']
        labels = ['subgroup alive 1', 'subgroup alive 2', 'subgroup dead 1', 'subgroup dead 2']
    
    for label, marker, color, group in zip(labels, markers, colors, np.unique(data[label_column])):
        if selected_groups == None or group in selected_groups:
            if plot_2d:
                ax.scatter(data[data[label_column] == group]['pc_'+str(axis[0])]
                           , data[data[label_column] == group]['pc_'+str(axis[1])]
                           , c=color
                           , marker=marker
                           , s=MARK_SIZE
                           , label=label
                           , edgecolors='k')
            else:
                ax.scatter(data[data[label_column] == group]['pc_'+str(axis[0])]
                           , data[data[label_column] == group]['pc_'+str(axis[1])]
                           , data[data[label_column] == group]['pc_'+str(axis[2])]
                           , c=color
                           , marker=marker
                           , label=label
                           , edgecolor='k'
                           , s=MARK_SIZE)
    ax.legend()

    plt.show()

#plot_decomposition(data, plot_kmeans=True, selected_groups=[0,1,2,3], axis=(1,2,3))
#plot_decomposition(data, plot_kmeans=True, selected_groups=[1,5], axis=(1,2,3))
#plot_decomposition(data, plot_kmeans=True, selected_groups=[6,7,8,9,10,11], axis=(1,2,3))
plot_decomposition(data, plot_2d=True, plot_kmeans=False, selected_groups=[0,1], axis=(1,2))
plot_decomposition(data, plot_2d=True, plot_kmeans=True, selected_groups=[1,2], axis=(1,2))
plot_decomposition(data, plot_2d=True, plot_kmeans=True, selected_groups=[1,3], axis=(1,2))
plot_decomposition(data, plot_2d=True, plot_kmeans=True, selected_groups=[1,4], axis=(1,2))
plot_decomposition(data, plot_2d=True, plot_kmeans=True, selected_groups=[2,3], axis=(1,2))
plot_decomposition(data, plot_2d=True, plot_kmeans=True, selected_groups=[2,4], axis=(1,2))
plot_decomposition(data, plot_2d=True, plot_kmeans=True, selected_groups=[3,4], axis=(1,2))
#plot_decomposition(data, plot_2d=True, plot_kmeans=False, selected_groups=[0,1,2,3,4], axis=(1,2))
