import pandas as pd
import os
from scipy import io
import shutil
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


dir_base = './t-sne/'
csv_list = os.listdir(dir_base+'csv/')

for i in range(int(len(csv_list)/2)):

    file_1 = csv_list[2*i]
    file_2 = csv_list[2*i+1]

    ## file 1
    data = pd.read_csv(dir_base+'csv/{}'.format(file_1), index_col = 0)
    data_y = data[['2RFS']]
    del data['2RFS']
    feature = data

    model = TSNE(learning_rate = 100)
    transformed = model.fit_transform(feature)

    xs = transformed[:,0]
    ys = transformed[:,1]

    k =data_y.values

    # # TSNE
    # plt.subplot(1, 2, 2)
    # plt.scatter(xs, ys, c=data_y.values, cmap= plt.cm.get_cmap('rainbow',2), alpha = 0.5)
    # plt.colorbar(ticks=range(2), format='recur %d')
    #
    name = file_1.split('_')
    fold = name[1].split('(')[1]
    # plt.title('{} ({} - {}'.format(name[0], fold, name[2]))

    ## file 2
    data = pd.read_csv(dir_base + 'csv/{}'.format(file_2), index_col=0)
    data_y = data[['2RFS']]
    del data['2RFS']
    feature = data

    # # TSNE
    # model = TSNE(learning_rate=100)
    # transformed = model.fit_transform(feature)
    #
    # xs = transformed[:, 0]
    # ys = transformed[:, 1]
    #
    # k = data_y.values
    #
    # plt.subplot(1, 2, 1)
    # plt.scatter(xs, ys, c=data_y.values, cmap=plt.cm.get_cmap('rainbow', 2), alpha=0.5)
    #
    name = file_2.split('_')
    fold = name[1].split('(')[1]
    # plt.title('{} ({} - {}'.format(name[0], fold, name[2]))
    #
    # plt.savefig(dir_base+'({} {}.png'.format(fold, name[0]))
    # plt.close()


    if name[0] != 'intra':
        print('intra 아니라 넘어감 !')
        continue

    if name[0] == 'intra' and fold == 'B)':
        print('intra-B fold라 넘어감 !')
        continue

    # scatter plot matrix
    data = pd.read_csv(dir_base+'csv/{}'.format(file_2), index_col = 0)
    palette_pair = {1:"royalblue", 0:"salmon"}
    sns.pairplot(data, hue='2RFS', palette=palette_pair)
    plt.savefig(dir_base+'[pairplot-train] ({} {}_{}.png'.format(fold, name[0],name[2]))
    plt.close()