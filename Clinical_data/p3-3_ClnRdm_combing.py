'''
[ 구현 내용 ] 
Matlab에서 clinical+radiomic feature로 classification 할 때 사용할 파일 생성
Output 폴더에 생성된 csv파일들을 모두 Matlab의 input으로 사용함
[ 내부 모듈 ]
▶ M_RadFeatures4Analysis : .mat 형태로 되어있는 radiomic feature들을 전처리
▶ M_ClnFeatures4Analysis : clinical feature들을 training/test set으로 나눔
'''

import os
import numpy as np
import pandas as pd
import M_RadFeatures4Analysis as F4A
import M_ClnFeatures4Analysis as CF4A
import sklearn

"""
1. 환자번호 : DW_Patients, Index_patients
2. Fold정보 : DW_Patients
3. feature 정보 : 
3. Radiomic feature : DW_Radiomic
4. Radiomic feature normalization : 
5. Fold, mm 별 Singificant feature 정보, Classifier 정보 : DW_Root
6. 
"""
meta_loop = ['peri','comb'];#, 'comb', 'cln']
meta_loop_peri = ['3mm','12mm']
meta_loop_comb = ['6mm','9mm']
Folds_main = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}  # [1,2,3,4,5]

date = '210412'

DW_Root = pd.read_csv(".\csv\Clinical_radiomic\i_ref_A.csv")
csv_path = "./csv/Clinical_radiomic/output({})/".format(date)
csv_name = csv_path+"{}_testfold({})_{}.csv" # intra, A, train, rf

for i, type in enumerate(meta_loop):
    print("*")
    for j, fold in enumerate(Folds_main.keys()):
        if type == 'intra' :
            featNum = DW_Root.groupby('type').get_group(type)['feature'].iloc[j]
            clf = DW_Root.groupby('type').get_group(type)['clf'].iloc[j]
            df_clf = pd.DataFrame(data=[clf], index=['clf'], columns=['clf'])

            cIntraRadFeat = F4A.Feature()
            cIntraClnFeat = CF4A.ClnFeature(date)
            intraRadFeat_train, intraRadFeat_test = cIntraRadFeat.sequence(type, fold, featNum)
            intraClnFeat_train, intraClnFeat_test = cIntraClnFeat.road_clnFeature(fold)
            intraCRFeat_train = pd.concat([intraClnFeat_train, intraRadFeat_train], axis=1)
            intraCRFeat_test = pd.concat([intraClnFeat_test, intraRadFeat_test], axis=1)

            intraCRFeat_train.to_csv(csv_name.format(type, fold, 'train'), index=True)
            intraCRFeat_test.to_csv(csv_name.format(type, fold, 'test'), index=True)

        elif type == 'peri' :
            for k, mm in enumerate(meta_loop_peri):
                typemm = type+mm
                featNum = DW_Root.groupby('type').get_group(typemm)['feature'].iloc[j]
                clf = DW_Root.groupby('type').get_group(typemm)['clf'].iloc[j]
                df_clf = pd.DataFrame(data=[clf], index=['clf'], columns=['clf'])

                cPeriRadFeat = F4A.Feature()
                cPeriClnFeat = CF4A.ClnFeature(date)
                periRadFeat_train, periRadFeat_test = cPeriRadFeat.sequence(typemm, fold, featNum)
                periClnFeat_train, periClnFeat_test = cPeriClnFeat.road_clnFeature(fold)
                periCRFeat_train = pd.concat([periClnFeat_train, periRadFeat_train], axis=1)
                periCRFeat_test = pd.concat([periClnFeat_test, periRadFeat_test], axis=1)

                periCRFeat_train.to_csv(csv_name.format(typemm, fold, 'train'), index=True)
                periCRFeat_test.to_csv(csv_name.format(typemm, fold, 'test'), index=True)

        elif type == 'comb' :
            for k, mm in enumerate(meta_loop_comb):
                typemm = type + mm
                featNum = DW_Root.groupby('type').get_group(typemm)['feature'].iloc[j]
                clf = DW_Root.groupby('type').get_group(typemm)['clf'].iloc[j]
                df_clf = pd.DataFrame(data=[clf], index=['clf'], columns=['clf'])

                cCombRadFeat = F4A.Feature()
                cCombClnFeat = CF4A.ClnFeature(date)
                combRadFeat_train, combRadFeat_test = cCombRadFeat.sequence(typemm, fold, featNum)
                combClnFeat_train, combClnFeat_test = cCombClnFeat.road_clnFeature(fold)
                combCRFeat_train = pd.concat([combClnFeat_train, combRadFeat_train], axis=1)
                combCRFeat_test = pd.concat([combClnFeat_test, combRadFeat_test], axis=1)

                combCRFeat_train.to_csv(csv_name.format(typemm, fold, 'train'), index=True)
                combCRFeat_test.to_csv(csv_name.format(typemm, fold, 'test'), index=True)


        elif type == 'cln':
            cIntraClnFeat = CF4A.ClnFeature(date)
            ClnFeat_train, ClnFeat_test = cIntraClnFeat.road_clnFeature(fold)

            # 2RFS
            cIntraRadFeat = F4A.Feature()
            RadFeat_train, RadFeat_test = cIntraRadFeat.sequence('intra', fold, 1)

            ClnFeat_train = pd.concat([ClnFeat_train, RadFeat_train['2RFS']], axis=1)
            ClnFeat_test = pd.concat([ClnFeat_test, RadFeat_test['2RFS']], axis=1)

            ClnFeat_train.to_csv(csv_name.format(type, fold, 'train'), index=True)
            ClnFeat_test.to_csv(csv_name.format(type, fold, 'test'), index=True)

#
# cClnFeature = CF4A.ClnFeature
# cln_train, cln_test = cClnFeature.road_clnFeature(cClnFeature,_test_fold='A')
#
# Feature = F4A.Feature()
# hcf_g1_raw, hcf_g2_raw = Feature.rawFeatures(_type='intra')
# hcf_g1, hcf_g2 = Feature.SplitandScale(hcf_g1_raw, hcf_g2_raw,'A')
# feature_rank = Feature.featureRank('A')
# feature_train, feature_test = Feature.Selection(feature_rank, 2)
#
# feature_train_X = feature_train; feature_test_X = feature_test
# gt_train_Y = feature_train['2RFS']; gt_test_Y = feature_test['2RFS']
# del feature_train_X['2RFS']; del feature_test_X['2RFS']
#
#
# trainset_cNr = pd.concat([cln_train, feature_train_X],axis=1)
# testset_cNr = pd.concat([cln_test, feature_test_X],axis=1)
#
#
# k=0
# #feature_train_X_s = scaler.fit_transform(feature_train_X)
# #feature_test_X_s = scaler.fit_transform(feature_test_X)
#
# scaler = StandardScaler()
# clf = svm.SVC(kernel='rbf', gamma='scale',shrinking=False,random_state=42)
#
# # clf.fit(feature_train_X.values, gt_train_Y.values)
# # clf_predictions = list(clf.predict(feature_test_X.values))
# # print(clf.get_params)
# # TN, FP, FN, TP = sklearn.metrics.confusion_matrix(list(gt_test_Y.values), clf_predictions, labels=[0,1]).ravel()
# # result_score = {'SEN':(TP/(TP+FN))*100, 'SPEC':(TN/(TN+FP))*100, 'PPV':(TP/(TP+FP))*100, 'NPV':(TN/(TN+FN))*100}
# # print("Accuracy: {}%".format(clf.score(feature_test_X.values, gt_test_Y.values) * 100))
# # print("SEN:{} \nSPEC:{}\nPPV:{}\nNPV:{}".format(result_score['SEN'], result_score['SPEC'], result_score['PPV'], result_score['NPV']))
# #
# # a=1
