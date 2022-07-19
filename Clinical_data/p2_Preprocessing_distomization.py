'''
Clinical feature cleansing code
'''
__author__='soominlee@swu.ac.kr'

import pandas as pd
import numpy as np
import openpyxl as xl
import os
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
import sklearn.preprocessing as sklearn_pp
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold

date = '210730'

## pandas print option
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

'''
1. clinical information 불러오기
2. clinical information 정제하기
    1. Missing value (10%미만) 처리 (10%미만은 이미 제외되어있는 파일을 사용함)
    2. Outlier elimination using z-score (numerical)
    3. Imputation 
    4. Min-max normalization (numerical)
    5. Convert to numerical(Dummy) variable (cateogircal)
    6. Elimination near zero variance
3. 결과 출력하기
[ref] https://dandyrilla.github.io/2017-08-12/pandas-10min/
'''

'''
Step 1. clinical information load
'''
df = pd.read_csv(".\csv\i_clinical_info_217_dtm.csv")

''' clinical feature 데이터프레임으로 선언 '''
dataNum = df['data_num']; RFS_2y = df['2yRFS']; fold = df['fold']                    # df에서 인덱스와 정답 분리
recurNmonth = df.loc[:,['2yRFS','month']]
fold.index = dataNum
recurNmonth.index = dataNum
RFS_2y.index = dataNum

del df['data_num']; del df['2yRFS']; del df['month'];                   # 해당 행 삭제
del df['fold']
df = pd.DataFrame(df.values, index=dataNum, columns=df.columns)         # clinical factor 데이터프레임 선언
df.index.name = "patient_No"                                            # 인덱스 이름 지정

categorical = pd.DataFrame(df.iloc[:,0:24])                             # categorical variable
numerical = pd.DataFrame(df.iloc[:,24:])                                # numerical variable

'''
Step 2. Outlier elimination using z-score (numerical)
        변수명: numerical_2
'''

numerical_2 = pd.DataFrame()
for i, col in enumerate(numerical.columns):
    X = numerical.loc[:, col]
    X_mean = np.nanmean(X, axis=0, dtype=float)
    X_std = np.nanstd(X, axis=0, dtype=float)

    # calculate Z-score
    Z = (X - X_mean)/X_std
    Z = pd.DataFrame(Z.T, index=dataNum, columns=[col])
    del_index = Z[(abs(Z[col])>2)].index.values.tolist()

    Y = pd.DataFrame(X, index=dataNum, columns=[col])
    for j, del_i in enumerate(del_index):
        Y.loc[del_i] = np.NaN

    if Y.isnull().sum().item()-X.isnull().sum().item() != len(del_index):
        print("[Error] X's null num + delete index num =/= Y's null num")
        print("\t {}, X_{}, Y_{}, del_{}".format(col,X.isnull().sum().item(),Y.isnull().sum().item(),len(del_index)))
        raise ValueError

    numerical_2 = pd.concat([numerical_2, Y], axis=1)
del i, j, del_i, col, X, X_mean, X_std, Z, del_index, Y

'''
Step 3. NaN value imputation
        - numerical variable : Multiple Imputation (sklearn library)
        - categorical variable : imputing as mode value
        변수명: numerical_3, categorical_3
'''

numerical_3 = pd.DataFrame(KNNImputer(n_neighbors=5, weights='distance').fit_transform(numerical_2),
                           index=dataNum, columns=numerical_2.columns)
categorical_3 = categorical
for col in categorical:
    categorical_3_recur_T = categorical_3[(RFS_2y.isin([1]))]
    categorical_3_recur_F = categorical_3[(RFS_2y.isin([0]))]
    mode_val_recur_T = categorical_3_recur_T[col].describe()['top']
    mode_val_recur_F = categorical_3_recur_F[col].describe()['top']

    #categorical_3[col].fillna(mode_val, inplace=True)
    categorical_3_recur_T[col].fillna(mode_val_recur_T, inplace=True)
    categorical_3_recur_F[col].fillna(mode_val_recur_F, inplace=True)

    categorical_3 = pd.concat([categorical_3_recur_T,categorical_3_recur_F], axis=0)
    categorical_3 = categorical_3.sort_index()
del col, mode_val_recur_T, mode_val_recur_F, categorical_3_recur_T, categorical_3_recur_F

'''
Step 4. Min-max normalization
        변수명: numerical_4
'''
scaler = sklearn_pp.MinMaxScaler()
numerical_4 = scaler.fit_transform(numerical_3)
numerical_4 = pd.DataFrame(numerical_4, index=dataNum, columns=numerical_3.columns)


'''
Step 5. Convert to numerical(Dummy) variable
        - 순서가 있는 범주형 데이터: T-stage, N-stage, Residual Tumor, tumor_stage
        - 순서가 없는 범주형 데이터 중 중복값을 갖는 데이터: pathology-location, anatomic_organ_subdivision
        변수명: categorical_4
'''
# 순서 있는 범주형 데이터 처리
categorical_4_order = categorical_3.iloc[:, 0:15]

map_T = {'T1':1, 'T2':2, 'T3':3}    # primary_tumor_pathologic_spread
map_N = {'N0':1, 'N1':2, 'N2':3}    # lymphnode_pathologic_spread
map_R = {'R0':1, 'R1':2, 'RX':1}    # residual_tumor
map_stage = {'StageIA':1, 'StageIB':2, 'StageIIA':3, 'StageIIB':4, 'StageIIIA':5}
map_dtm = {'High': 2, 'Normal':1, 'Low':0}

categorical_4_order['primary_tumor_pathologic_spread'] = categorical_4_order['primary_tumor_pathologic_spread'].map(map_T)
categorical_4_order['lymphnode_pathologic_spread'] = categorical_4_order['lymphnode_pathologic_spread'].map(map_N)
categorical_4_order['residual_tumor'] = categorical_4_order['residual_tumor'].map(map_R)
categorical_4_order['tumor_stage'] = categorical_4_order['tumor_stage'].map(map_stage)

for i in range (11):
    i = i+4
    categorical_4_order.iloc[:, i] = categorical_4_order.iloc[:, i].map(map_dtm)

scaler = sklearn_pp.MinMaxScaler()
categorical_4_order_columns = categorical_4_order.columns
categorical_4_order = scaler.fit_transform(categorical_4_order)
categorical_4_order = pd.DataFrame(categorical_4_order, index=dataNum, columns=categorical_4_order_columns)
#categorical_4_order = pd.get_dummies(categorical_4_order, prefix_sep='#', drop_first=False)

del map_T, map_N, map_R

# 순서 없는 범주형 데이터 처리 (중복값 X)
categorical_4_wo_order = categorical_3[categorical_3.columns.difference(categorical_4_order_columns)]

double_val = ["pathology-location"]

categorical_4_double = categorical_4_wo_order[:][double_val]
categorical_4_wo_order = categorical_4_wo_order.drop(["pathology-location"], axis=1)
categorical_4_wo_order = pd.get_dummies(categorical_4_wo_order, prefix_sep='#', drop_first=True)

del double_val

# 순서 없는 범주형 데이터 처리 (중복값O)
categorical_4_double = pd.get_dummies(categorical_4_double, prefix_sep='#')

for i, col in enumerate(categorical_4_double):
    if ',' in col:
        X = col.split('#')[1]
        double_name = X.split(',')
        double_true = categorical_4_double[(categorical_4_double[col]==1)].index.values.tolist()
        for j, idx in enumerate(double_true):
            f_name = col.split('#')[0]
            f_name_1 = f_name+'#'+double_name[0]
            f_name_2 = f_name+'#'+double_name[1]
            categorical_4_double.loc[idx,f_name_1] = 1
            categorical_4_double.loc[idx,f_name_2] = 1
            categorical_4_double
        del categorical_4_double[col]

del i, col, X, double_name, double_true, j, idx, f_name, f_name_1, f_name_2

# 새로 생성된 column 전처리
categorical_4_double = categorical_4_double.fillna(0)
categorical_4_double = categorical_4_double.astype(int)

categorical_4 = pd.concat([categorical_4_order, categorical_4_wo_order, categorical_4_double], axis=1)
del categorical_4_order, categorical_4_wo_order, categorical_4_double

## 순서있는 범주형 dummy화 했을 때
# categorical_4 = pd.concat([categorical_4_wo_order, categorical_4_double], axis=1)
# del categorical_4_wo_order, categorical_4_double

clinical_4 = pd.concat([categorical_4, numerical_4],axis=1)
clinical_4.to_csv('./csv/o_preprocessed_clinical_information_({})_defore NZV.csv'.format(date), index=True)


'''
Step 5. Eliminate near-zero variance varible
        변수명: numerical_5, categorical_5
'''
# near zero variance feature remove function
def nzv(th, features_df, row_idx):
    selector = VarianceThreshold(th)
    after_nzv = selector.fit_transform(features_df)
    idx_nzv_n = selector.get_support(indices=True)          # number 형태의 index
    idx_nzv = [list(features_df)[idx] for idx in idx_nzv_n] # 이름이 있는 index
    features_nzv = pd.DataFrame(after_nzv, index=row_idx, columns=idx_nzv)
    print("●near zero variacne features with threshold({}): {}".format(str(th),str(len(features_df.columns)-len(features_nzv.columns))))
    print("○removed features : ",set(list(features_df.columns))-set(list(features_nzv.columns)))


    return features_nzv

# # delte near zero variance
categorical_5 = nzv(0.07, categorical_4, dataNum)           # categorical variables
numerical_5 = nzv(0.03, numerical_4, dataNum)               # numerical variables

'''
결과 저장
'''
path = '.\csv\{}.xlsx'
name='o_preprocessed_clinical_information({})_dummy'.format(date)

preprocessed = pd.concat([categorical_5, numerical_5], axis=1)
preprocessed.to_excel(path.format(name), index=True)
preprocessed.to_csv('.\csv\{}.csv'.format(name), index=True)

corr = categorical_5.corr()

import seaborn as sns
cmap = sns.light_palette("darkgray", as_cmap=True)
sns.heatmap(corr, annot=True, cmap=cmap)
plt.show()

numerical_5_corr = numerical_5 #numerical_5.drop(['Hct', 'ANC'],axis='columns')
categorical_5_corr = categorical_5 #categorical_5.drop(['lymphnode_pathologic_spread'],axis='columns')

preprocessed_corr = pd.concat([categorical_5_corr, numerical_5_corr, fold, RFS_2y], axis=1)
preprocessed_corr.index = dataNum
preprocessed_corr.to_csv('.\csv\{}_corr.csv'.format(name), index=True)

#
# '''
# 추가) 엑셀에 feature별로 sheet에 저장 (numerical)
# 추가) 엑셀에 version 별로 sheet에 저장 (categorical)
# '''
#
# path = '.\csv\{}.xlsx'
# name1 = 'o_numerical_variable(210105)'
# name2 = 'o_categorical_variable(210105)'
# numerical_col = ['original', 'outlier', 'imputation', 'MM norm', 'NZV(Fin.)']
#
# if os.path.isfile(path.format(name1)):
#     os.remove(path.format(name1))
# if os.path.isfile(path.format(name2)):
#     os.remove(path.format(name2))
#
# for i, col in enumerate(numerical.columns):
#     numerical_set = pd.DataFrame()
#     try:
#         numerical_set = pd.concat([numerical[col], numerical_2[col], numerical_3[col], numerical_4[col], numerical_5[col]], axis=1)
#         numerical_set.columns = numerical_col
#     except:
#         numerical_set = pd.concat([numerical[col], numerical_2[col], numerical_3[col], numerical_4[col]], axis=1)
#         numerical_set.columns = numerical_col[:-1]
#         print('[Error] at save as EXCEL) 삭제된 feature.')
#
#
#     if not os.path.exists(path.format(name1)):
#         with pd.ExcelWriter(path.format(name1), mode='w', engine='openpyxl') as writer:
#             numerical_set.to_excel(writer, index=True, sheet_name=col)
#     else:
#         with pd.ExcelWriter(path.format(name1), mode='a', engine='openpyxl') as writer:
#             numerical_set.to_excel(writer, index=True, sheet_name=col)
#
# categorical_ver = ['original', 'imputation', 'Dummy', 'NZV(Fin.)']
#
# writer = pd.ExcelWriter(path.format(name2), mode='w', engine='openpyxl')
# categorical.to_excel(writer, index=True, sheet_name='original')
# writer.mode = 'a'
# categorical_3.to_excel(writer, index=True, sheet_name='outlier')
# categorical_4.to_excel(writer, index=True, sheet_name='dummpy')
# categorical_5.to_excel(writer, index=True, sheet_name='NZV(Fin.)')