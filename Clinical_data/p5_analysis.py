import os
import pandas as pd
from scipy import io
import glob
import shutil

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def forClnRadComb():
    csv_list = os.listdir('./t-sne/csv/')

    for a in range(int(len(csv_list)/2)):
        file = csv_list[2*a]

        data = pd.read_csv('./t-sne/csv/{}'.format(file), index_col=0)
        y_data = data[['2RFS']]
        del data['2RFS']
        feature= data

        name = file.split('_')
        crnt_fold = name[1].split('(')[1]; crnt_fold = crnt_fold.split(')')[0]
        clf = name[0]

        dir_mat = dir_base.format('mat/{}'.format(clf))
        mat_list = os.listdir('{}/'.format(dir_mat))

        crnt_mat_name = ""
        try:
           crnt_mat_name = mat_list.index("{}_testfold-{}_svm.mat".format(clf, crnt_fold))
        except:
            crnt_mat_name = mat_list.index("{}_testfold-{}_rf.mat".format(clf, crnt_fold))
        crnt_mat_name = mat_list[crnt_mat_name]
        crnt_mat = io.loadmat('{}/{}'.format(dir_mat, crnt_mat_name))

        crnt_clf = crnt_mat_name.split('_')[2]; crnt_clf = crnt_clf.split('.')[0]

        if crnt_clf == 'svm':
            crnt_mat = pd.concat([pd.DataFrame(crnt_mat['test_gt']),
                                  pd.DataFrame(crnt_mat['label_{}_temp'.format(crnt_clf)]),
                                  pd.DataFrame(crnt_mat['score_{}_temp'.format(crnt_clf)])], axis=1)
        elif crnt_clf == 'rf':
            crnt_mat = pd.concat([pd.DataFrame(crnt_mat['test_gt']),
                                  pd.DataFrame(crnt_mat['label_{}_temp'.format(crnt_clf)][0, 0]),
                                  pd.DataFrame(crnt_mat['score_{}_temp'.format(crnt_clf)][0, 0])], axis=1)
        crnt_mat.columns = ['gt', 'label', 'score_nonrecur', 'score_recur']

        crnt_patients_info = patient_info[patient_info['Fold'] == crnt_fold]
        crnt_mat.index = crnt_patients_info.index

        predicted_label = []

        gt = crnt_mat.loc[:, 'gt']
        label = crnt_mat.loc[:, 'label']

        for k in range(len(crnt_patients_info)):
            if (gt.iloc[k] == label.iloc[k]) and (gt.iloc[k] == 2):
                predicted_label.append('TP')  # TP
            if (gt.iloc[k] == label.iloc[k]) and (gt.iloc[k] == 1):
                predicted_label.append('TN')  # TN
            if (gt.iloc[k] != label.iloc[k]) and (gt.iloc[k] == 2):
                predicted_label.append('FN')  # FN
            if (gt.iloc[k] != label.iloc[k]) and (gt.iloc[k] == 1):
                predicted_label.append('FP')  # FP

        predicted_label_df = pd.DataFrame(predicted_label)
        predicted_label_df.index = crnt_mat.index
        predicted_label_df.columns = ['predicted_label']

        ## TSNE
        # model = TSNE(learning_rate=100)
        # transformed = model.fit_transform(feature)
        #
        # xs = transformed[:, 0]
        # ys = transformed[:, 1]
        #
        # plt.scatter(xs, ys, c=predicted_label_df.values, cmap=plt.cm.get_cmap('rainbow', 4), alpha=0.5)
        # cbar = plt.colorbar(ticks=range(4), format='%d')
        #
        #
        # plt.title('{} {}'.format(clf, crnt_fold))
        #
        # #plt.savefig('./t-sne/{} {}.png'.format(clf, crnt_fold))
        # plt.close()


        ## pairplot
        if clf != 'cln' and clf != 'intra':
            print('intra 아니라 넘어감 !')
            continue

        if clf == 'intra' and crnt_fold == 'B':
            print('intra-B fold라 넘어감 !')
            continue
        print('{}  {} '.format(clf, crnt_fold))

        feature_TPFN = pd.concat([feature, predicted_label_df], axis=1)
        sns.pairplot(feature_TPFN, hue='predicted_label', palette=palette_pair)
        # plt.show()
        plt.savefig('./t-sne/[pairplot] {} {}.png'.format(crnt_fold, clf))
        plt.close()

def forRad():
    dir_base = 'E:/research/Prediction/NSCLC/Codes/img_G217/{}'
    csv_list_tsne = os.listdir('./t-sne/csv/')
    csv_list_pred_rslt = [_ for _ in os.listdir(dir_base.format('.')) if _.endswith('.csv')]

    for a in range(int(len(csv_list_tsne)/2)):
        file = csv_list_tsne[2*a]

        data = pd.read_csv('./t-sne/csv/{}'.format(file), index_col=0)
        y_data = data[['2RFS']]
        del data['2RFS']
        feature= data

        name = file.split('_')
        crnt_fold = name[1].split('(')[1]; crnt_fold = crnt_fold.split(')')[0]
        clf = name[0]

        crnt_csv_idx = ""
        try:
            if clf == 'intra':
                tmp_clf = 'intratumoral'
            if clf == 'peri3mm' or clf == 'peri12mm':
                tmp_clf = clf.replace('peri','peritumoral')
            if clf == 'comb6mm' or clf == 'comb9mm':
                tmp_clf = clf.replace('comb', 'combine')
                tmp_clf
            if clf == 'cln':
                continue
            print(tmp_clf)
            crnt_csv_idx = csv_list_pred_rslt.index("{}_{}_score.csv".format(crnt_fold, tmp_clf))
        except:
            print("파일 없음 error")

        crnt_csv_name = csv_list_pred_rslt[crnt_csv_idx]
        crnt_csv = pd.read_csv(dir_base.format(crnt_csv_name))

        crnt_patients_info = patient_info[patient_info['Fold'] == crnt_fold]
        crnt_csv.index = crnt_patients_info.index

        predicted_label = []

        gt = y_data.iloc[:,0] + 1
        label = crnt_csv.iloc[:, 0]

        for k in range(len(crnt_patients_info)):
            print('{}nd  gt:{}, predicted:{}'.format(k, gt.iloc[k], label.iloc[k]))
            if (gt.iloc[k] == label.iloc[k]) and (gt.iloc[k] == 2):
                predicted_label.append('TP')  # TP
            if (gt.iloc[k] == label.iloc[k]) and (gt.iloc[k] == 1):
                predicted_label.append('TN')  # TN
            if (gt.iloc[k] != label.iloc[k]) and (gt.iloc[k] == 2):
                predicted_label.append('FN')  # FN
            if (gt.iloc[k] != label.iloc[k]) and (gt.iloc[k] == 1):
                predicted_label.append('FP')  # FP

        predicted_label_df = pd.DataFrame(predicted_label)
        predicted_label_df.index = crnt_csv.index
        predicted_label_df.columns = ['predicted_label']

        ## TSNE
        # model = TSNE(learning_rate=100)
        # transformed = model.fit_transform(feature)
        #
        # xs = transformed[:, 0]
        # ys = transformed[:, 1]
        #
        # plt.scatter(xs, ys, c=predicted_label_df.values, cmap=plt.cm.get_cmap('rainbow', 4), alpha=0.5)
        # cbar = plt.colorbar(ticks=range(4), format='%d')
        #
        #
        # plt.title('{} {}'.format(clf, crnt_fold))
        #
        # #plt.savefig('./t-sne/{} {}.png'.format(clf, crnt_fold))
        # plt.close()


        ## pairplot
        if clf != 'cln' and clf != 'intra':
            print('intra 아니라 넘어감 !')
            continue

        if clf == 'intra' and crnt_fold == 'B':
            print('intra-B fold라 넘어감 !')
            continue
        print('{}  {} '.format(clf, crnt_fold))

        feature_TPFN = pd.concat([feature, predicted_label_df], axis=1)
        sns.pairplot(feature_TPFN, hue='predicted_label', palette=palette_pair)
        plt.savefig('./t-sne/only-radio [pairplot] {} {}.png'.format(crnt_fold, clf))
        plt.close()


if __name__ == '__main__':

    dir_base = 'E:/research/Prediction/NSCLC/Codes/img_G217_210204/{}'
    patient_info = pd.read_csv(dir_base.format('patient_list.csv'), index_col='Data_num')

    '''
        primary loop : clinical > intratumoral > peritumoral (3mm, 12mm) > combined (6mm, 9mm)
            secondary loop : A fold > B fold > C fold > D fold > E fold
                (third loop) : Group 1 > Group 2 > Group 3
    '''

    loop_prm = ['cln', 'intra', 'peri3mm', 'peri12mm', 'comb6mm', 'comb9mm']
    loop_scnd = ['A', 'B', 'C', 'D', 'E']
    palette_pair = {"TP": "royalblue", "TN": "salmon", "FP": "pink", "FN": "lightsteelblue"}

    forClnRadComb()
    forRad()
