cd('E:/research/Prediction/NSCLC/Codes/') % sumin

num_of_data = size(datainfo, 1);

dir_base = 'E:/research/Prediction/NSCLC/';
 
% 배열을 담을 cell 선언
minmax_r_cell_p = cell(num_of_data, 2);  
minmax_c_cell_p = cell(num_of_data, 2);   
windowing_cell = cell(num_of_data, 3);  
stat_cell = cell(num_of_data, 3);   


for data_num = 1:num_of_data
    disp(data_num)
    display(num2str(data_num));  % 몇번째 데이터인지 출력
    display(datainfo(data_num,1));    % n번째 데이터에 첫번째 열을 출력..?
    data_id = num2str(datainfo{data_num,1});    % 데이터 아이디 할당
    data_id_ori = datainfo{data_num,1};
    %temp = strsplit(datainfo{data_num,1},' ');
    %data_id = temp{2};
    
    %temp = strsplit(data_id,'_');
    %dada_id_MR = temp{1}
    
    %% DICOM
    str = [dir_base, 'Data/', data_id, '/'];
    tmp = dir(str);
    cd(str);
    
    temp_dir=dir('*.dcm'); % 폴더 내 모든 dcm
    disp(temp_dir)
    img_info=dicominfo(temp_dir(1).name); % 첫 번째 dcm 이용, temp_dir(1).name -> filename
    
    %disp(img_info)
    
    img_spec=cell(1,21);
    img_spec{1}=data_num;
    img_spec{2}=data_id;
    img_spec{3}=img_info.StudyDate;
    % img_spec{4}=img_info.ConvolutionKernel;
    img_spec{5}=img_info.PatientBirthDate;
    img_spec{6}=img_info.PatientSex;
    img_spec{7}=img_info.PixelSpacing;
    img_spec{8}=img_info.SliceThickness;
    img_spec{9}=img_info.WindowCenter;
    img_spec{10}=img_info.WindowWidth;
    img_spec{11}=img_info.RescaleIntercept;
    img_spec{12}=img_info.RescaleSlope;

    % Reading 3D image
    img_3d=zeros([512,512,size(temp_dir,1)]); % size(temp_dir,1) -> slice number
    for slicenum=1:size(img_3d,3)
        img_3d(:,:,size(img_3d,3)-slicenum+1)=dicomread(temp_dir(slicenum).name);
    end
    clear 'slicenum'

    slicenum = datainfo{data_num, 3};
    disp(size(slicenum))
    img_raw = double(dicomread(temp_dir(slicenum).name));
    
    
    %% MASK
    % Reading 3D mask
    mask_3d = zeros(size(img_3d));
    mask_3d_dilated = zeros(size(img_3d));
    
    %Intratumoral
    str=[dir_base, 'Data/mask/', data_id '/'];
    cd(str);
    
    temp_dir=dir('*.png');
    
    ss = size(mask_3d,3);
    for slicenum=1:size(mask_3d,3)
        mask_3d(:,:,slicenum)=mask_3d(:,:,slicenum)+...
            double(imread(temp_dir(slicenum).name));
    end
    
    slicenum = datainfo{data_num, 3};
    mask = double(imread(temp_dir(slicenum).name));
     
    [row, col] = find(mask);
    
    if strcmp(flg,'intra')==0
        %Dilated
        str=[dir_base, strcat('Data/mask_dilated(',mm,')/'), data_id '/'];
        cd(str);

        temp_dir=dir('*.png');

        ss = size(mask_3d_dilated,3);
        for slicenum=1:size(mask_3d_dilated,3)
            mask_3d_dilated(:,:,slicenum)=mask_3d_dilated(:,:,slicenum)+...
                double(imread(temp_dir(slicenum).name));
        end

        %Peritumoral
        slicenum = datainfo{data_num, 3};
        mask_dilated = double(imread(temp_dir(slicenum).name)); 
        [row, col] = find(mask_dilated);

    end
    
    min_r = min(row); 
    min_c = min(col); 
    
    max_r = max(row); 
    max_c = max(col); 

    minmax_r_cell_p{data_num, 1} = min_r; 
    minmax_r_cell_p{data_num, 2} = max_r; 
    minmax_c_cell_p{data_num, 1} = min_c; 
    minmax_c_cell_p{data_num, 2} = max_c; 
    
    windowing_cell{data_num, 1} = data_id;
    windowing_cell{data_num, 2} = img_info.WindowCenter(1);
    windowing_cell{data_num, 3} = img_info.WindowWidth(1);
 

    stat_cell{data_num, 1} = data_id;
    stat_cell{data_num, 2} = mean(img_raw(:));
    stat_cell{data_num, 3} = std(img_raw(:));
 
    wc = img_info.WindowCenter(1);
    ww = img_info.WindowWidth(1);
   
    %% intensity rescaling (HU->GR)
    n_wc = -600; 
    n_ww = 1500; 
    
    img_raw = img_raw * img_info.RescaleSlope + img_info.RescaleIntercept;
    img_norm=hu2norm_window(img_raw,n_wc,n_ww);     % -1350~150HU를 0~1사이로 변환
    img = uint8(round(255*img_norm));               % 0~255로 변환
    
    % Save img
    path_log = [dir_base, 'Codes/Log/', date];
    str=[path_log, '/img/'];
   
    if(isfolder(str)==0)
        mkdir(str);
    end    
    
    str=[str, data_id,'_', num2str(slicenum),'_img.png'];

    
    imwrite(uint8(img),str,'png');
    
    % Save files
    str=[dir_base, 'Codes/Log/', date, '/mat_p/',flg,'_',mm,'/'];
    
    if(isfolder(str)==0)
        mkdir(str);
    end
    
    str=[dir_base, 'Codes/Log/', date, '/mat_p/',flg,'_',mm,'/',num2str(data_num),'.mat'];
    
    if strcmp(flg,'intra')==0
        mask_peri = mask_dilated - mask;      %%peri
        save(str,'img_raw','img','mask', 'mask_peri');
    else
        save(str,'img_raw','img','mask');
    end
        
    %% PATCH
    img_patch = img(min_r-10:max_r+10, min_c-10:max_c+10); 
    img_raw_patch = img_raw(min_r-10:max_r+10, min_c-10:max_c+10);
    mask_patch = mask(min_r-10:max_r+10, min_c-10:max_c+10); 
    
    if strcmp(flg,'intra')==0
        mask_patch_dilated = mask_dilated(min_r-10:max_r+10, min_c-10:max_c+10);  %%peri
    end
    
    % Save imgr
    group = 0;
    if(datainfo{data_num, 2} == 1)
        group = 1;
    end
    str = [dir_base, 'Codes/Log/', date, '/img_G/'];
    if(isfolder(str)==0)
        mkdir(str);
    end
    
    str=[dir_base, 'Codes/Log/', date, '/img_G/',num2str(group),'_', data_id, '_', num2str(slicenum),'_img_patch.png'];
    
    imwrite(uint8(img_patch),str,'png');
    
    % Save files
    str=[dir_base, 'Codes/Log/', date, '/mat_p_patch/',flg,'_',mm];
    
    if(isfolder(str)==0)
        mkdir(str);
    end
    
    
    str=[dir_base, 'Codes/Log/', date, '/mat_p_patch/',flg,'_',mm,'/',num2str(data_num),'.mat'];
    if strcmp(flg,'intra')==0
        mask_patch_peri = mask_patch_dilated - mask_patch             %%peri
        save(str,'img_raw_patch','img_patch','mask_patch', 'mask_patch_peri'); 
    else
        save(str,'img_raw_patch','img_patch','mask_patch');
    end
    clear 'str'
    clear 'tmp'
    
end
cd('E:/research/Prediction/NSCLC/Codes/')
