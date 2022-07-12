% flg = 'intra'; mm_arr = ["0mm"];
% flg = 'peri'; mm_arr = ["12mm"];
% flg = 'comb'; mm_arr = ["3mm","6mm","9mm","12mm","15mm","18mm","21mm","24mm","27mm","30mm"];
flg = 'peri'; mm_arr = ["3mm"];

cd('E:/research/Prediction/NSCLC/Codes/')
addpath('./Function') 
for j_mm=1:1
%   
    date = '210728' %% 백업을 위한 실행 날짜 입력
%     mm = char(mm_arr(j_mm));
%     RBRB = strcat('mm-->', mm);
%     %mwb.Update(1,1,i/8,RBRB,[0.7,0,0.5]);
%     
%     %mwb.Update(2,1,1/5,'progress',[0,0.2,0.5]);    
%     test_p_1_1_imread;
%     disp('image read done!')
% 
%     %mwb.Update(2,1,2/5,'progress',[0,0.2,0.5]);
%     test_p_1_5_save_2group;
%     disp('divide into group done!')

%     %mwb.Update(2,1,3/5,'progress',[0,0.2,0.5]);                    
%     test_p_2_2_save_HCF_2group;
%     disp('save HCF done!')
    
  
    gpu = gpuDeviceCount;
    gpuDevice(1);
    result_save_path = ['mat_5_1_five-fold-cv_',date,'/'];  
	sheet=j_mm;
    %% classification_peritumoral
    %mwb.Update(2,1,4/5,'progress',[0,0.2,0.5]);
	test_p_3_1_classify_HCF_2_cv_210514;
	test_excel;
%     
end
% 