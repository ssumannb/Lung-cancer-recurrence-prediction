import M_compare_auc_delong_xu
import pandas as pd
import numpy as np
import os

from scipy import io

def mat2csv():
    path = './ROC/comparing/cln-rad_prediction/mat/'
    osList_fold = os.listdir(path)

    for i, fold in enumerate(osList_fold):
        path = './ROC/comparing/cln-rad_prediction/mat/{}/'.format(fold)
        pList = pd.read_csv('./ROC/comparing/cln-rad_prediction/pList_217_size.csv', index_col='data_num')
        osList_file = os.listdir(path)
        fold_cv = ['A', 'B', 'C', 'D', 'E']

        for j, file in enumerate(osList_file):
            pred = pd.DataFrame()

            tmp_splt = file.split('_')[2]
            tmp_splt = tmp_splt.split('.')[0]

            assert fold_cv[j] in file, 'fold 불일치'

            pList_cv = pList[pList['fold']==(j+1)]

            mat_score = pd.DataFrame()
            mat_gt = pd.DataFrame(io.loadmat(path+file)['test_gt'])
            name = 'score_{}_temp'.format(tmp_splt)
            if tmp_splt == 'svm':
                mat_score = pd.DataFrame(io.loadmat(path+'/'+file)[name]).iloc[:,1]
            elif tmp_splt == 'rf':
                mat_score = pd.DataFrame(io.loadmat(path+'/'+file)[name]).iloc[0,0]
                mat_score = pd.DataFrame(mat_score).iloc[:,1]

            mat = pd.concat([mat_score, mat_gt], axis=1)
            pred = pd.concat([pred, mat], axis=0)
            pred.index = pList_cv.index
            pred = pd.concat([pred, pList_cv], axis=1)
            pred = pred.sort_values(by=['fold', 'RFS'])
            del pred[0]
            pred.rename(columns={1: 'prob'}, inplace=True)

            path_dst = './ROC/comparing/cln-rad_prediction_sizegroup/{}'.format(fold)
            if not os.path.exists(path_dst):
                os.mkdir(path_dst)
            pred.to_csv(path_dst+'/test_score_{}_{}.csv'.format(fold_cv[j], fold))




if __name__ == '__main__':
    #mat2csv()


    rad_root = pd.DataFrame()
    cNr_root = pd.DataFrame()

    clfs = ['intra', 'peri3mm', 'peri12mm', 'comb6mm', 'comb9mm']

    for i, clf in enumerate(clfs):
        fold_list_cv = ['A','B','C','D','E']
        print('*********{}*********'.format(clf))
        # clinical and radiomic List
        path_radcln = './ROC/comparing/cln-rad_prediction_sizegroup/{}'.format(clf)
        radcln_List = os.listdir(path_radcln)
        # radiomic List
        path_rad = './ROC/comparing/radiomic_prediction/{}'.format(clf)
        rad_List = os.listdir(path_rad)
        results = []

        '''total'''
        # rad_total = pd.read_csv('./ROC/comparing/radiomic_prediction/total/{}.csv'.format(clf))
        # radcln_total = pd.read_csv(path_radcln+'.csv', index_col='data_num')
        #
        # gt = np.array(radcln_total['RFS'].tolist())
        # rad_pred_total = rad_total['prob'].to_numpy(copy=False)
        # radcln_pred_total = radcln_total['prob'].to_numpy(copy=False)
        #
        # auc, var = M_compare_auc_delong_xu.delong_roc_variance(gt, rad_pred_total)
        # print('auc - rad ' + str(round(auc,2)))
        # auc, var = M_compare_auc_delong_xu.delong_roc_variance(gt, radcln_pred_total)
        # print('auc - radcln ' + str(round(auc,2)))
        # result_cv = M_compare_auc_delong_xu.delong_roc_test(gt, rad_pred_total, radcln_pred_total)
        # print(result_cv)

        '''fold'''
        for i, fold_cv in enumerate(fold_list_cv):
            print('--fold {}--'.format(fold_cv))
            assert fold_cv in radcln_List[i], 'radiomic and clinical fold 불일치'
            assert fold_cv in rad_List[i], 'radiomic fold 불일치'

            radcln = pd.read_csv(path_radcln+'/'+radcln_List[i], index_col='data_num')
            rad = pd.read_csv(path_rad+'/'+rad_List[i], header=None)
            rad_gt = pd.read_csv('./ROC/comparing/radiomic_prediction/test_gt_{}fold.csv'.format(fold_cv), header=None)
            rad.columns = ['prob']
            rad_gt.columns = ['gt']
            rad_gt = rad_gt - 1


            rad.index, rad_gt.index = radcln.index, radcln.index

            idx_G1 = radcln['size'] < 3.0
            idx_G2 = (3.0 <= radcln['size']) & (radcln['size'] < 5.0)
            idx_G3 = radcln['size'] > 5.0

            #G1
            gt_G1 = np.array(rad_gt[idx_G1]['gt'].tolist())
            rad_pred_G1 = rad[idx_G1]['prob'].to_numpy(copy=False)
            radcln_pred_G1 = radcln[idx_G1]['prob'].to_numpy(copy=False)

            #G2
            gt_G2 = np.array(rad_gt[idx_G2]['gt'].tolist())
            rad_pred_G2 = rad[idx_G2]['prob'].to_numpy(copy=False)
            radcln_pred_G2 = radcln[idx_G2]['prob'].to_numpy(copy=False)

            #G3
            gt_G3 = np.array(rad_gt[idx_G3]['gt'].tolist())
            rad_pred_G3 = rad[idx_G3]['prob'].to_numpy(copy=False)
            radcln_pred_G3 = radcln[idx_G3]['prob'].to_numpy(copy=False)

            # gt = np.array(rad_gt['gt'].tolist())
            # radcln_pred = radcln['prob'].to_numpy(copy=False)
            # rad_pred = rad['prob'].to_numpy(copy=False)

            result_cv_G1 = M_compare_auc_delong_xu.delong_roc_test(gt_G1, rad_pred_G1, radcln_pred_G1)
            print("Group 1 : {}".format(result_cv_G1))
            result_cv_g2 = M_compare_auc_delong_xu.delong_roc_test(gt_G2, rad_pred_G2, radcln_pred_G2)
            print("Group 2 : {}".format(result_cv_g2))
            result_cv_G3 = M_compare_auc_delong_xu.delong_roc_test(gt_G3, rad_pred_G3, radcln_pred_G3)
            print("Group 3 : {}".format(result_cv_G3))

        #     results.append(result_cv[0][0])
        #
        # print('>> mean '+str(sum(results)/len(results)))