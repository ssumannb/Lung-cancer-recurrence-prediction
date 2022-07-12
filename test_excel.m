filename = ['E:/research/Prediction/NSCLC/Codes/', result_save_path, flg,'_',mm,'.xlsx'];

xls_svm = {};
xls_rf = {};
xls_esbl = {};
fold = ["A","B","C","D","E"];
criteria = ["acc","sen","spe","ppv","npv","AUC"];

if strcmp(flg, 'intra')
    num = 2:69; num = transpose(num);
elseif strcmp(flg, 'peri')
    num = 2:58; num = transpose(num);
elseif strcmp(flg, 'comb')
    num = 2:127; num = transpose(num) 
end

classifier = {strcat(flg,'_',mm),'R','a','n','d','o','m'};
xlRange = 'A1';

for i=0:4
    temp = [criteria; cv_result_rf{5-i,1}];
    num2 = [strcat('**(test)',fold(i+1),'-fold');num];
    temp2 = [num2,temp];
    xls_rf = [xls_rf; temp2];
end
xls_rf=[classifier;xls_rf];

classifier = {'','S','','V','','M',''};
for i=0:4
    temp = [criteria; cv_result_svm{5-i,1}];
    num2 = [strcat('**',fold(i+1),'-fold');num];
    temp2 = [num2,temp];
    xls_svm = [xls_svm; temp2];
end
xls_svm=[classifier;xls_svm];

classifier = {'','e','','s','','bl',''};
for i=0:4
    temp = [criteria; cv_result_esbl{5-i,1}];
    num2 = [strcat('**',fold(i+1),'-fold');num];
    temp2 = [num2,temp];
    xls_esbl = [xls_esbl; temp2];
end
xls_esbl=[classifier;xls_esbl];

xls_result = [xls_rf, xls_svm, xls_esbl];
xlswrite(filename,xls_result,sheet,xlRange)

clearvars -except 'mm' 'flg' 'datainfo' 'mwb' 'j_mm' 'mm_arr'