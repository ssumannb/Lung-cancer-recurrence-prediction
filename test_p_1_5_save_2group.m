cd('E:/research/Prediction/NSCLC/Codes/')
num_of_data = size(datainfo, 1);
dir_base = ['./Log/',date,'/mat_p_patch/'];

%%
num_g1 = 0;
num_g2 = 0;

% 각 그룹 g1,g2에 해당하는 데이터 갯수 count
for data_num = 1:num_of_data    
    if(datainfo{data_num, 2} == 0) % recurrence X
        num_g1 = num_g1 + 1;
    end
    if(datainfo{data_num, 2} == 1) % recurrence O
        num_g2 = num_g2 + 1;
    end
end


%% Create cell arrays  (g1, g2 size의 array생성)
img_g1=cell(num_g1,1);
img_raw_g1=cell(num_g1,1);
mask_g1=cell(num_g1,1);
mask_peri_g1=cell(num_g1,1);
name_g1=zeros(num_g1,1);

img_g2=cell(num_g2,1);
img_raw_g2=cell(num_g2,1);
mask_g2=cell(num_g2,1);
mask_peri_g2=cell(num_g2,1);
name_g2=zeros(num_g2,1);

cnt1 = 1;
cnt2 = 1;

for data_num = 1:num_of_data
    display(num2str(data_num));
    str=[dir_base, flg,'_',mm,'/',num2str(data_num),'.mat'];
    
    % NSCLC\Codes\mat_p_patch-sumin\(datanum).mat파일의 각 행렬 값에 
    % img 행렬, raw img행렬, mask행렬 이 할당되어있음
    
    if strcmp(flg,'intra') == 0
        load(str,'img_patch','img_raw_patch','mask_patch', 'mask_patch_peri');
    else
        load(str,'img_patch','img_raw_patch','mask_patch');
    end
    clear 'str'

    % 데이터 로드해서 그룹별로 나눠줌
    if(datainfo{data_num, 2} == 0)
        img_g1{cnt1}=img_patch;
        img_raw_g1{cnt1}=img_raw_patch;
        mask_g1{cnt1}=mask_patch;
        if strcmp(flg,'intra')==0
            mask_peri_g1{cnt1}=mask_patch_peri;
        end
        name_g1(cnt1)=datainfo{data_num, 1};
        cnt1 = cnt1 + 1;
    end
    
    if(datainfo{data_num, 2} == 1)
        img_g2{cnt2}=img_patch;
        img_raw_g2{cnt2}=img_raw_patch;
        mask_g2{cnt2}=mask_patch;
        if strcmp(flg,'intra')==0
            mask_peri_g2{cnt2}=mask_patch_peri;
        end
        name_g2(cnt2)=datainfo{data_num, 1};
        cnt2 = cnt2 + 1;
    end 

    clear 'img_patch' 'img_raw_patch' 'mask_patch'  'mask_patch_peri'
end

clear 'data_num' 'cnt1' 'cnt2'


str = ['./Log/',date,'/mat_p_cell/', flg,'_',mm];
if(isfolder(str)==0)
    mkdir(str);
end

str_1=[str,'/g1.mat'];
str_2=[str,'/g2.mat'];

if strcmp(flg,'intra') == 0
    save(str_1,'img_g1','img_raw_g1','mask_g1','mask_peri_g1','name_g1');  %combination peritumoral
    save(str_2,'img_g2','img_raw_g2','mask_g2','mask_peri_g2','name_g2');  %combination peritumoral
else
    save(str_1,'img_g1','img_raw_g1','mask_g1','name_g1'); 
    save(str_2,'img_g2','img_raw_g2','mask_g2','name_g2');
end


clear 'str' 'str_1' 'str_2'