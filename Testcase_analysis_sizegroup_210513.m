cd('E:/research/Prediction/NSCLC/Codes/')
dir_base = './mat_5_3_testcase_sizegroup_analysis/data/';
dir_base_ROC = './mat_5_3_testcase_sizegroup_analysis/ROC/';   %% ROC 커브 그릴때 용이하도록 csv파일로 변환해주는 폴더
dir_base_KM = './mat_5_3_testcase_sizegroup_analysis/Kaplan-meier_csv/'; %% KM 커브 그릴때 용이하도록 csv파일로 변환해주는 폴더
% mm_arr = ["9mm"] %,"6mm","9mm","12mm","15mm","18mm","21mm","24mm","27mm","30mm"];
% flg = 'comb';
% mm_arr = ["6mm","9mm"];
mm_arr = ["0mm"]; flg = 'intra';  % intratumoral 
% flg = 'cln'; % clinical
%%%%    str=[dir_base, 'Codes/mat_p_patch/',num2str(data_num),'.mat'];
% flg = 'peri';
% mm_arr = ["3mm","12mm"];

%xlswrite(filename,xls_result,sheet,xlRange)
xls_Group1 = {};
xls_Group2 = {};
xls_Group3 = {};
col = ["clf","mm","fold","grp","acc","sen","spe","ppv","npv","AUC"];
xls_Group1 = [xls_Group1;col];
xls_Group2 = [xls_Group2;col];
xls_Group3 = [xls_Group3;col];
test_month_km = [datainfosg.month];
test_gt_km = [datainfosg.recurr];    

for mm_iter=1:1
    mm= char(mm_arr(mm_iter))
    str=[dir_base,flg,'/',mm,'/'];   % fold별 bestcase result 정리해 놓은 위치
    list = dir(str); n=length(list);
    test_label_km = {}; test_label_km = ["kaplan"; test_label_km];
    
    
    for f_iter=3:n 
        file_path = strcat(str,'/',list(f_iter).name);
        load(file_path)
        file_name = list(f_iter).name;

        % A_classifiy_HCF_random_comb3mm_svm.mat -> fold, classifier, rf iteration idex
        tmp = split(file_name,'.'); splt = split(tmp(1), '_'); splt2 = split(splt(2),'-');
        clear 'tmp' 
        fold = cell2mat(splt2(2)); clf = cell2mat(splt(3)); rf_idx = 1;
        
        if strcmp(clf,'rf')
            test_label_rf = test_label_rf();
            test_score_rf = test_score_rf();
            test_score_roc = test_score_rf(:,2);
        end
        if strcmp(clf,'svm')
            total_result = compute_accuracy2_180514(test_label_svm,test_score_svm,Fold_test_gt);
            test_label_svm = test_label_svm();
            test_score_svm = test_score_svm();
            test_score_roc = test_score_svm(:,2);
        end
        if strcmp(clf,'esbl')
            total_result = compute_accuracy2_180514(test_label_esbl,test_prob_esbl,Fold_test_gt);
            test_label_esbl = test_label_esbl();
            test_prob_esbl = test_prob_esbl();
            test_score_roc = test_prob_esbl(:,2);
        end
        
        filename_roc = [dir_base_ROC, mm,'/test_score_',fold,'_', flg, '.csv'];        
        writematrix(test_score_roc, filename_roc);
        
        cwf = datainfosg.fold==fold; 
        cwf = datainfosg(cwf,:);
        
        % group 1
        G1_idx = cwf.tumor_size<3.0; G1_idx = cwf(G1_idx, :); 
        Group1 = table2array(G1_idx(:,2)); Group1_gt = table2array(G1_idx(:,4))+1;
        % group 2
        G2_idx = (3.0<=cwf.tumor_size & cwf.tumor_size<5.0); G2_idx = cwf(G2_idx,:);
        Group2 = table2array(G2_idx(:,2)); Group2_gt = table2array(G2_idx(:,4))+1;
        % group 3
        G3_idx = 5.0<=cwf.tumor_size; G3_idx = cwf(G3_idx,:);
        Group3 = table2array(G3_idx(:,2)); Group3_gt = table2array(G3_idx(:,4))+1;
        
        clear 'G1_idx' 'G2_idx' 'G3_idx'
        if strcmp(clf,'svm')
            disp('svm!')
            G1_label = test_label_svm(Group1, 1); G1_score = test_score_svm(Group1,:);
            G2_label = test_label_svm(Group2, 1); G2_score = test_score_svm(Group2,:);
            G3_label = test_label_svm(Group3, 1); G3_score = test_score_svm(Group3,:);
        end

        if strcmp(clf,'rf')
            disp('rf!')
            G1_label = test_label_rf(Group1,1); G1_score = test_score_rf(Group1,:);
            G2_label = test_label_rf(Group2,1); G2_score = test_score_rf(Group2,:);
            G3_label = test_label_rf(Group3,1); G3_score = test_score_rf(Group3,:);
        end

        
        if strcmp(clf,'esbl')
            disp('esbl!')
            G1_label = test_label_esbl(Group1,1); G1_score = test_prob_esbl(Group1,:);
            G2_label = test_label_esbl(Group2,1); G2_score = test_prob_esbl(Group2,:);
            G3_label = test_label_esbl(Group3,1); G3_score = test_prob_esbl(Group3,:);
        end
        
        tag = [string(flg) string(mm) string(fold)]; 
        G1_eval = compute_accuracy2_180514(G1_label, G1_score, Group1_gt); xls_Group1 = [xls_Group1; [tag,'G1',G1_eval]];
        G2_eval = compute_accuracy2_180514(G2_label, G2_score, Group2_gt); xls_Group2 = [xls_Group2; [tag,'G2',G2_eval]];
        G3_eval = compute_accuracy2_180514(G3_label, G3_score, Group3_gt); xls_Group3 = [xls_Group3; [tag,'G3',G3_eval]];
        
        G1_eval;
        G2_eval;
        G3_eval;
        
        a=0;
    end
    filename_km = [dir_base_KM, flg, '_', mm, '.csv'];

    xls_Group1 = [xls_Group1; ["-","-","-","-"," "," "," "," "," "," "]];
    xls_Group1 = [xls_Group1; ["-","-","-","-"," "," "," "," "," "," "]];
    xls_Group2 = [xls_Group2; ["-","-","-","-"," "," "," "," "," "," "]];
    xls_Group2 = [xls_Group2; ["-","-","-","-"," "," "," "," "," "," "]];
    xls_Group3 = [xls_Group3; ["-","-","-","-"," "," "," "," "," "," "]];
    xls_Group3 = [xls_Group3; ["-","-","-","-"," "," "," "," "," "," "]];

end
xls_result = [xls_Group1; xls_Group2; xls_Group3];
filename=[dir_base,'sizeGroup_analysis-',flg,'.xlsx'];
xlswrite(filename,xls_result,1,'A1');
a = 1;
% [X_rf,Y_rf,T_svm,AUC_svm] = perfcurve(Group3_test_gt, Group3_score_svm(:,2), '2');
% plot(X_rf,Y_rf)