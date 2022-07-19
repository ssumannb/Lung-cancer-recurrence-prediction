import pandas as pd
import numpy as np
import seaborn as sns

import os
from scipy import io
from matplotlib import colors
import matplotlib.pyplot as plt


path_mat = './csv/Radiomic feature/{}'
list_dir = os.listdir(path_mat.format('mat/'))

feature_name_intra = pd.read_csv(path_mat.format('feature_intra.csv'), index_col=False)
feature_name_peri = pd.read_csv(path_mat.format('feature_peri.csv'))
feature_name_comb = pd.read_csv(path_mat.format('feature_comb.csv'))
cm = sns.diverging_palette(240, 10, as_cmap=True)


mms = ["3mm","6mm","9mm","12mm","15mm","18mm","21mm","24mm","27mm","30mm"]
folds = ['E','D','C','B','A']

mmBYfold = []
# mmBYfold_style = []
for j, mm in enumerate(mms):
    for i, fold in enumerate(folds):
        tmp = '{}_{}'.format(mm, fold)
        mmBYfold = mmBYfold + ['flag'] +[tmp]
        # mmBYfold_style = mmBYfold_style + [tmp]

comb_feature_weight_csv = pd.DataFrame()
comb_feature_weight_sum_csv = pd.DataFrame()
intra_feature_weight_csv = pd.DataFrame()
intra_feature_weight_sum_csv = pd.DataFrame()
peri_feature_weight_csv = pd.DataFrame()
peri_feature_weight_sum_csv = pd.DataFrame()
def best_featurenum(df):
    auc = df['AUC'].round(4)
    row = df[auc==auc.max()]

    if row.shape[0] != 1:
        acc = row['ACC']
        row = row[acc==acc.max()]

    if row.shape[0] != 1:
        row = row.drop(row.index[1:], axis=0)
        row

    idx = row.index[0]

    return row, idx

def draw_color_cell(x,color):
    color = f'background-color:{color}'
    return color

# def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
#     rng = M - m
#     norm = colors.Normalize(m - (rng * low),
#                             M + (rng * high))
#     normed = norm(s.values)
#     c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
#     return ['background-color: %s' % color for color in c]

for i, file in enumerate(list_dir):
    feature_weights = pd.DataFrame(io.loadmat(path_mat.format('mat/' + file))['feat_weights_nca'])

    criteria = ['ACC','SEN','SPEC','PPV','NPV','AUC']
    #results_rf = np.split(results_rf, 6)
    best_feat_num_list = []
    print(file)
    for j, fold in enumerate(folds):
        # feature_weights = pd.DataFrame(io.loadmat(path_mat.format('mat/' + file))['feat_weights_nca'])
        results_svm = pd.DataFrame(io.loadmat(path_mat.format('mat/' + file))['cv_result_svm'][j][0])
        results_rf = pd.DataFrame(io.loadmat(path_mat.format('mat/' + file))['cv_result_rf'][j][0])
        results_svm.columns = criteria; results_rf.columns = criteria;

        # 최강자전 가리기ㅋ^ㅁ^ best performance feature number selection
        best_svm, best_svm_idx = best_featurenum(results_svm)
        best_rf, best_rf_idx = best_featurenum(results_rf)

        best_tmp = pd.concat([best_rf, best_svm])
        best_tmp.index = [best_rf_idx, best_svm_idx]
        best_of_best, best_of_best_idx = best_featurenum(best_tmp)
        best_of_best_idx += 2
        best_feat_num_list.append(best_of_best_idx)
        #if (idx_auc_svm.shape[0]!=1)

    #best_feat_weight_idx = []
    feature_weight_style = feature_weights
    feature_num_tmp = pd.DataFrame()
    for j, fold in enumerate(folds):
        best_feat_weight_idx = []
        feature_weight_fold = feature_weights.iloc[:,j]
        tmptmp = np.zeros_like(feature_weight_fold)
        best_feat_weight_list = feature_weight_fold.nlargest(best_feat_num_list[j])
        best_feat_weight_idx.append(best_feat_weight_list.index)

        for k, idx in enumerate(best_feat_weight_idx):
            tmptmp[idx] = 1
        tmptmp = pd.DataFrame(tmptmp)
        feature_num_tmp = pd.concat([feature_num_tmp,tmptmp, feature_weight_fold], axis=1)
        feature_num_tmp
        #feature_weight_style.style.background_gradient(subset=[j], cmap='WhGnRd')


    if (feature_weights.shape[0] == 127):
        feature_weights.index = feature_name_comb
        # feature_weights_sum = feature_weights.sum(axis=1)
        # feature_weights_sum.index = feature_name_comb

        comb_feature_weight_csv = pd.concat([comb_feature_weight_csv, feature_num_tmp], axis=1)
        #comb_feature_weight_sum_csv = pd.concat([comb_feature_weight_sum_csv, feature_weights_sum], axis=1)

        # comb_feature_weight_csv.style.background_gradient(cmap='WhGnRd')

    elif (feature_weights.shape[0] == 69):
        feature_weights.index == feature_name_intra
        # feature_weights_sum = feature_weights.sum(axis=1)
        # feature_weights_sum.index = feature_name_intra

        intra_feature_weight_csv = pd.concat([intra_feature_weight_csv, feature_num_tmp], axis=1)
        #intra_feature_weight_sum_csv = pd.concat([intra_feature_weight_sum_csv, feature_weights_sum], axis=1)

    elif (feature_weights.shape[0] == 58):
        feature_weights.index == feature_name_peri
        # feature_weights_sum = feature_weights.sum(axis=1)
        # feature_weights_sum.index = feature_name_peri

        peri_feature_weight_csv = pd.concat([peri_feature_weight_csv, feature_num_tmp], axis=1)
        #peri_feature_weight_sum_csv = pd.concat([peri_feature_weight_sum_csv, feature_weights_sum], axis=1)

        # even_range = np.max([np.abs(peri_feature_weight_csv.values.min()), np.abs(peri_feature_weight_csv.values.max())])
        # peri_feature_weight_csv = peri_feature_weight_csv.style.apply(background_gradient,
        #                cmap=cm,
        #                m=-even_range,
        #                M=even_range).set_precision(2)

    else:
        print('feature number incorrect error')
        break;



comb_feature_weight_csv.columns = mmBYfold
# comb_feature_weight_sum_csv.columns = mms
# intra_feature_weight_csv.columns = folds
# intra_feature_weight_sum_csv.columns = ['intra']
peri_feature_weight_csv.columns = mmBYfold
# peri_feature_weight_sum_csv.columns = mms
#
# comb_feature_weight_csv.to_excel(path_mat.format('comb_weight.xlsx'), index=True)
# comb_feature_weight_sum_csv.to_csv(path_mat.format('comb_Sum_weight.csv'), index=True)
intra_feature_weight_csv.to_excel(path_mat.format('intra_weight.xlsx'), index=True)
# intra_feature_weight_sum_csv.to_csv(path_mat.format('intra_Sum_weight.csv'), index=True)
# peri_feature_weight_csv = peri_feature_weight_csv.style.background_gradient(subset=mmBYfold_style, cmap=cm)

# peri_feature_weight_csv.to_excel(path_mat.format('peri_weight.xlsx'), index=True)
# peri_feature_weight_sum_csv.to_csv(path_mat.format('peri_Sum_weight.csv'), index=True)