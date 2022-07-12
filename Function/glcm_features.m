% function [f1,f6,f8,f9,f11]=glcm_features(glcm,N)
function [f1,f2,f6,f7,f8,f9,f11]=glcm_features(glcm,N)

glcm=glcm/sum(glcm(:));  % glcm을 구성하고 있는 요소들을 개수->퍼센티지로 변환
% f1: angular second moment
f1=sum(sum(glcm.^2));
% f2: contrast
temp=abs(repmat((1:N),N,1)-repmat((1:N)',1,N));
f2=sum(sum(temp.^2.*glcm));
clear 'temp'
% f6: sum average
temp=repmat((1:N),N,1)+repmat((1:N)',1,N);
f6=sum(sum(temp.*glcm));
% clear 'temp'
% f7: sum variance
f7=sum(sum((temp-f6).^2.*glcm));
clear 'temp'
% f8: sum entropy
temp=repmat((1:N),N,1)+repmat((1:N)',1,N);
sump=zeros(2*N-1,1);
for num=1:length(sump)
    ind=find(temp==(num+1));
    sump(num)=sum(glcm(ind));
end
clear 'temp' 'num' 'ind'
f8=-sum(sump.*log(sump+eps));
clear 'sump'
% f9: entropy
temp=glcm.*log(glcm+eps);
f9=-sum(sum(temp));
clear 'temp'
% f11 difference entropy
temp=abs(repmat((1:N),N,1)-repmat((1:N)',1,N));
diffp=zeros(N,1);
for num=1:length(diffp)
    ind=find(temp==(num-1));
    diffp(num)=sum(glcm(ind));
end
clear 'temp' 'num' 'ind'
f11=-sum(diffp.*log(diffp+eps));
clear 'diffp'
