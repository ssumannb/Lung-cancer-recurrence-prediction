import os
import pandas as pd
from scipy import io
import glob
import shutil

def byFold():
    for i, clf in enumerate(loop_prm):
        print('{}'.format(clf))
        dir_mat = dir_base.format('mat/{}'.format(clf))
        mat_list = os.listdir('{}/'.format(dir_mat))
        for j, fold in enumerate(loop_scnd):
            print('- {} fold'.format(fold))
            crnt_file = mat_list[j].split('.')[0]
            crnt_fold = (crnt_file.split('_')[1]).split('-')[1]
            crnt_clf = crnt_file.split('_')[2]
    
            crnt_mat = io.loadmat('{}/{}'.format(dir_mat,crnt_file))
            if crnt_clf == 'svm':
                crnt_mat = pd.concat([pd.DataFrame(crnt_mat['test_gt']),
                                      pd.DataFrame(crnt_mat['label_{}_temp'.format(crnt_clf)]),
                                      pd.DataFrame(crnt_mat['score_{}_temp'.format(crnt_clf)])], axis = 1)
            elif crnt_clf =='rf':
                crnt_mat = pd.concat([pd.DataFrame(crnt_mat['test_gt']),
                                      pd.DataFrame(crnt_mat['label_{}_temp'.format(crnt_clf)][0,0]),
                                      pd.DataFrame(crnt_mat['score_{}_temp'.format(crnt_clf)][0,0])], axis=1)
            crnt_mat.columns = ['gt', 'label', 'score_nonrecur', 'score_recur']
            assert crnt_fold == fold, 'fold가 다릅니다'
    
            crnt_patients_info = patient_info[patient_info['Fold']==fold]
            crnt_mat.index = crnt_patients_info.index
    
    
            tumor_list = glob.glob(dir_base.format('{}-fold/*.png'.format(fold)))
            if not os.path.exists(dir_base.format('{}-fold/{}/'.format(fold, clf))): os.mkdir(dir_base.format('{}-fold/{}/'.format(fold, clf)))
            # if not os.path.exists(dir_base.format('{}-fold/TN/'.format(fold))): os.mkdir(dir_base.format('{}-fold/TN/'.format(fold)))
            # if not os.path.exists(dir_base.format('{}-fold/FP/'.format(fold))): os.mkdir(dir_base.format('{}-fold/FP/'.format(fold)))
            # if not os.path.exists(dir_base.format('{}-fold/FN/'.format(fold))): os.mkdir(dir_base.format('{}-fold/FN/'.format(fold)))
    
            for k, tumor in enumerate(tumor_list):
                file_name = tumor.split('\\')[1]
                tumor_num = int((tumor.split('\\')[1]).split('_')[2])
                gt = crnt_mat.loc[tumor_num,'gt']
                label = crnt_mat.loc[tumor_num, 'label']
                t_stg = crnt_patients_info.loc[tumor_num, 'T-stage']
                score = crnt_mat.loc[tumor_num, 'score_recur']
                file_name_new = '[{}]{}_{}_{}_prob=[{}].png'.format(clf ,str(gt-1), str(tumor_num).zfill(3),t_stg, score.round(3))
    
                TPFN = ['TP','TN','FN','FP']
                if (gt==label) and (gt==2):
                    tmp_path = '/{}-fold/{}/{}'.format(fold, clf, TPFN[0])
                    if not os.path.exists(dir_base.format(tmp_path)) : os.mkdir(dir_base.format(tmp_path))
                    shutil.copy(dir_base.format('/{}-fold/{}'.format(fold, file_name)),
                                dir_base.format('{}/{}'.format(tmp_path, file_name_new)))
                if (gt==label) and (gt==1):
                    tmp_path = '/{}-fold/{}/{}'.format(fold, clf, TPFN[1])
                    if not os.path.exists(dir_base.format(tmp_path)): os.mkdir(dir_base.format(tmp_path))
                    shutil.copy(dir_base.format('/{}-fold/{}'.format(fold, file_name)),
                                dir_base.format('{}/{}'.format(tmp_path, file_name_new)))
                if (gt!=label) and (gt==2):
                    tmp_path = '/{}-fold/{}/{}'.format(fold, clf, TPFN[2])
                    if not os.path.exists(dir_base.format(tmp_path)): os.mkdir(dir_base.format(tmp_path))
                    shutil.copy(dir_base.format('/{}-fold/{}'.format(fold, file_name)),
                                dir_base.format('{}/{}'.format(tmp_path, file_name_new)))
                if (gt!=label) and (gt==1):
                    tmp_path = '/{}-fold/{}/{}'.format(fold, clf, TPFN[3])
                    if not os.path.exists(dir_base.format(tmp_path)): os.mkdir(dir_base.format(tmp_path))
                    shutil.copy(dir_base.format('/{}-fold/{}'.format(fold, file_name)),
                                dir_base.format('{}/{}'.format(tmp_path, file_name_new)))
    
            tumor_list


def byGroup():
    for i, clf in enumerate(loop_prm):
        print('{}'.format(clf))
        dir_mat = dir_base.format('mat/{}'.format(clf))
        mat_list = os.listdir('{}/'.format(dir_mat))
        for j, fold in enumerate(loop_scnd):
            print('- {} fold'.format(fold))
            crnt_file = mat_list[j].split('.')[0]
            crnt_fold = (crnt_file.split('_')[1]).split('-')[1]
            crnt_clf = crnt_file.split('_')[2]

            crnt_mat = io.loadmat('{}/{}'.format(dir_mat, crnt_file))
            if crnt_clf == 'svm':
                crnt_mat = pd.concat([pd.DataFrame(crnt_mat['test_gt']),
                                      pd.DataFrame(crnt_mat['label_{}_temp'.format(crnt_clf)]),
                                      pd.DataFrame(crnt_mat['score_{}_temp'.format(crnt_clf)])], axis=1)
            elif crnt_clf == 'rf':
                crnt_mat = pd.concat([pd.DataFrame(crnt_mat['test_gt']),
                                      pd.DataFrame(crnt_mat['label_{}_temp'.format(crnt_clf)][0, 0]),
                                      pd.DataFrame(crnt_mat['score_{}_temp'.format(crnt_clf)][0, 0])], axis=1)
            crnt_mat.columns = ['gt', 'label', 'score_nonrecur', 'score_recur']
            assert crnt_fold == fold, 'fold가 다릅니다'

            crnt_patients_info = patient_info[patient_info['Fold'] == fold]
            crnt_mat.index = crnt_patients_info.index

            tumor_list = glob.glob(dir_base.format('{}-fold/*.png'.format(fold)))
            if not os.path.exists(dir_base.format('{}/'.format(clf))): os.mkdir(
                dir_base.format('{}/'.format(clf)))
            if not os.path.exists(dir_base.format('{}/GR_1'.format(clf))): os.mkdir(
                dir_base.format('{}/GR_1'.format(clf)))
            if not os.path.exists(dir_base.format('{}/GR_2'.format(clf))): os.mkdir(
                dir_base.format('{}/GR_2'.format(clf)))
            if not os.path.exists(dir_base.format('{}/GR_3'.format(clf))): os.mkdir(
                dir_base.format('{}/GR_3'.format(clf)))
            # if not os.path.exists(dir_base.format('{}-fold/TN/'.format(fold))): os.mkdir(dir_base.format('{}-fold/TN/'.format(fold)))
            # if not os.path.exists(dir_base.format('{}-fold/FP/'.format(fold))): os.mkdir(dir_base.format('{}-fold/FP/'.format(fold)))
            # if not os.path.exists(dir_base.format('{}-fold/FN/'.format(fold))): os.mkdir(dir_base.format('{}-fold/FN/'.format(fold)))

            for k, tumor in enumerate(tumor_list):
                file_name = tumor.split('\\')[1]
                tumor_num = int((tumor.split('\\')[1]).split('_')[2])
                gt = crnt_mat.loc[tumor_num, 'gt']
                label = crnt_mat.loc[tumor_num, 'label']
                t_stg = crnt_patients_info.loc[tumor_num, 'T-stage']
                score = crnt_mat.loc[tumor_num, 'score_recur']
                group = crnt_patients_info.loc[tumor_num, 'Group']
                file_name_new = '[{}]{}_{}_{}_prob=[{}].png'.format(fold, str(gt - 1), str(tumor_num).zfill(3), t_stg,
                                                                    score.round(3))

                TPFN = ['TP', 'TN', 'FN', 'FP']
                if (gt == label) and (gt == 2):
                    tmp_path = '{}/GR_{}/{}'.format(clf, group, TPFN[0])
                    if not os.path.exists(dir_base.format(tmp_path)): os.mkdir(dir_base.format(tmp_path))
                    shutil.copy(dir_base.format('/{}-fold/{}'.format(fold, file_name)),
                                dir_base.format('{}/{}'.format(tmp_path, file_name_new)))
                if (gt == label) and (gt == 1):
                    tmp_path = '{}/GR_{}/{}'.format(clf, group, TPFN[1])
                    if not os.path.exists(dir_base.format(tmp_path)): os.mkdir(dir_base.format(tmp_path))
                    shutil.copy(dir_base.format('/{}-fold/{}'.format(fold, file_name)),
                                dir_base.format('{}/{}'.format(tmp_path, file_name_new)))
                if (gt != label) and (gt == 2):
                    tmp_path = '{}/GR_{}/{}'.format(clf, group, TPFN[2])
                    if not os.path.exists(dir_base.format(tmp_path)): os.mkdir(dir_base.format(tmp_path))
                    shutil.copy(dir_base.format('/{}-fold/{}'.format(fold, file_name)),
                                dir_base.format('{}/{}'.format(tmp_path, file_name_new)))
                if (gt != label) and (gt == 1):
                    tmp_path = '{}/GR_{}/{}'.format(clf, group, TPFN[3])
                    if not os.path.exists(dir_base.format(tmp_path)): os.mkdir(dir_base.format(tmp_path))
                    shutil.copy(dir_base.format('/{}-fold/{}'.format(fold, file_name)),
                                dir_base.format('{}/{}'.format(tmp_path, file_name_new)))

            tumor_list

def byClf():
    for i, clf in enumerate(loop_prm):
        print('{}'.format(clf))
        dir_mat = dir_base.format('mat/{}'.format(clf))
        mat_list = os.listdir('{}/'.format(dir_mat))
        for j, fold in enumerate(loop_scnd):
            print('- {} fold'.format(fold))
            crnt_file = mat_list[j].split('.')[0]
            crnt_fold = (crnt_file.split('_')[1]).split('-')[1]
            crnt_clf = crnt_file.split('_')[2]

            crnt_mat = io.loadmat('{}/{}'.format(dir_mat, crnt_file))
            if crnt_clf == 'svm':
                crnt_mat = pd.concat([pd.DataFrame(crnt_mat['test_gt']),
                                      pd.DataFrame(crnt_mat['label_{}_temp'.format(crnt_clf)]),
                                      pd.DataFrame(crnt_mat['score_{}_temp'.format(crnt_clf)])], axis=1)
            elif crnt_clf == 'rf':
                crnt_mat = pd.concat([pd.DataFrame(crnt_mat['test_gt']),
                                      pd.DataFrame(crnt_mat['label_{}_temp'.format(crnt_clf)][0, 0]),
                                      pd.DataFrame(crnt_mat['score_{}_temp'.format(crnt_clf)][0, 0])], axis=1)
            crnt_mat.columns = ['gt', 'label', 'score_nonrecur', 'score_recur']
            assert crnt_fold == fold, 'fold가 다릅니다'

            crnt_patients_info = patient_info[patient_info['Fold'] == fold]
            crnt_mat.index = crnt_patients_info.index

            tumor_list = glob.glob(dir_base.format('{}-fold/*.png'.format(fold)))
            if not os.path.exists(dir_base.format('{}-fold/{}/'.format(fold, clf))): os.mkdir(
                dir_base.format('{}-fold/{}/'.format(fold, clf)))
            # if not os.path.exists(dir_base.format('{}-fold/TN/'.format(fold))): os.mkdir(dir_base.format('{}-fold/TN/'.format(fold)))
            # if not os.path.exists(dir_base.format('{}-fold/FP/'.format(fold))): os.mkdir(dir_base.format('{}-fold/FP/'.format(fold)))
            # if not os.path.exists(dir_base.format('{}-fold/FN/'.format(fold))): os.mkdir(dir_base.format('{}-fold/FN/'.format(fold)))

            for k, tumor in enumerate(tumor_list):
                file_name = tumor.split('\\')[1]
                tumor_num = int((tumor.split('\\')[1]).split('_')[2])
                gt = crnt_mat.loc[tumor_num, 'gt']
                label = crnt_mat.loc[tumor_num, 'label']
                t_stg = crnt_patients_info.loc[tumor_num, 'T-stage']
                score = crnt_mat.loc[tumor_num, 'score_recur']
                file_name_new = '[{}]{}_{}_{}_prob=[{}].png'.format(fold, str(gt - 1), str(tumor_num).zfill(3), t_stg,
                                                                    score.round(3))

                TPFN = ['TP', 'TN', 'FN', 'FP']
                if (gt == label) and (gt == 2):
                    tmp_path = '/{}/{}'.format(clf, TPFN[0])
                    if not os.path.exists(dir_base.format(tmp_path)): os.mkdir(dir_base.format(tmp_path))
                    shutil.copy(dir_base.format('/{}-fold/{}'.format(fold, file_name)),
                                dir_base.format('{}/{}'.format(tmp_path, file_name_new)))
                if (gt == label) and (gt == 1):
                    tmp_path = '/{}/{}'.format(clf, TPFN[1])
                    if not os.path.exists(dir_base.format(tmp_path)): os.mkdir(dir_base.format(tmp_path))
                    shutil.copy(dir_base.format('/{}-fold/{}'.format(fold, file_name)),
                                dir_base.format('{}/{}'.format(tmp_path, file_name_new)))
                if (gt != label) and (gt == 2):
                    tmp_path = '/{}/{}'.format(clf, TPFN[2])
                    if not os.path.exists(dir_base.format(tmp_path)): os.mkdir(dir_base.format(tmp_path))
                    shutil.copy(dir_base.format('/{}-fold/{}'.format(fold, file_name)),
                                dir_base.format('{}/{}'.format(tmp_path, file_name_new)))
                if (gt != label) and (gt == 1):
                    tmp_path = '/{}/{}'.format(clf, TPFN[3])
                    if not os.path.exists(dir_base.format(tmp_path)): os.mkdir(dir_base.format(tmp_path))
                    shutil.copy(dir_base.format('/{}-fold/{}'.format(fold, file_name)),
                                dir_base.format('{}/{}'.format(tmp_path, file_name_new)))

            tumor_list

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


    ## 실행부
    #byFold()
    #byGroup()
    byClf()