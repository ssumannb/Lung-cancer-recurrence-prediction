cd('E:/research/Prediction/NSCLC/Codes/')
dir_base = ['./Log/',date,'/mat_p_cell/'];


%% group 1
str = [dir_base, flg,'_',mm,'/g1.mat']; % str = [dir_base, 'Codes/mat_p_cell/g1.mat'];
load(str);


% img_gn : windowing 조절한 data
% img_raw_gn : 원본 data
% mask_gn : 마스크 png

if strcmp(flg,'peri') == 0
    intra=save_texture_features_original_190520(img_g1, img_raw_g1,mask_g1,1);           %tumor only & combination
end
if strcmp(flg,'intra') == 0
    peri=save_texture_features_original_190520(img_g1, img_raw_g1, mask_peri_g1, 1);   	 %peri only & combination
end


if strcmp(flg,'peri') == 0
    shape=save_shape_features_190130(img_g1,mask_g1,1);
    hcf=[intra, shape];
    if strcmp(flg,'comb')==1
        hcf_comb=[intra, peri, shape]; % combi
    end
    clear 'intra' 'shape'
else
    hcf_peri=[peri];   %perionly
    clear 'peri'
end


str = ['./Log/',date,'/mat_p_hcf/',flg,'_',mm,'/'];
if(isfolder(str)==0)
    mkdir(str);
end
str1=[str,'/hcf_g1.mat'];

if strcmp(flg,'peri') == 0
    save(str1,'hcf');
    if strcmp(flg,'comb')==1
        save(str1,'hcf_comb');
    end
else
    save(str1,'hcf_peri');
end

%% group 2
str = [dir_base,flg,'_',mm,'/g2.mat'];
load(str);

if strcmp(flg,'peri') == 0
    intra=save_texture_features_original_190520(img_g2, img_raw_g2, mask_g2,1);          %tumor only & combination
end
if strcmp(flg,'intra') == 0
    peri=save_texture_features_original_190520(img_g2, img_raw_g2, mask_peri_g2, 1);     %peri only & combination
end
%hcf=temp;
%hcf2=temp2;
%clear 'temp'

if strcmp(flg,'peri') == 0
    shape=save_shape_features_190130(img_g2,mask_g2,1);
    hcf=[intra, shape];
    if strcmp(flg,'comb')==1
        hcf_comb=[intra, peri, shape]; % combi
    end
    clear 'intra' 'shape'
else
    hcf_peri=[peri];   %perionly
    clear 'peri'
end


hold on

str = ['./Log/',date,'/mat_p_hcf/',flg,'_',mm];
if(isfolder(str)==0)
    mkdir(str);
end
str1=[str,'/hcf_g2.mat'];
if strcmp(flg,'peri') == 0
    save(str1,'hcf');
    if strcmp(flg,'comb')==1
        save(str1,'hcf_comb');
    end
else
    save(str1,'hcf_peri');
end
