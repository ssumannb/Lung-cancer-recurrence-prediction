function im_out = hu2norm_window(im_in,WindowCenter,WindowWidth)
val1=WindowCenter-floor(WindowWidth/2)+1;
val2=WindowCenter+floor(WindowWidth/2);
temp=double(im_in).*double(im_in>val1).*double(im_in<val2)+...
    val1*double(im_in<=val1)+val2*double(im_in>=val2);

im_out=double(temp-val1)/WindowWidth;

%function im_out = hu2norm_window(im_in,WindowCenter,WindowWidth,RescaleIntercept,RescaleSlope)
%val1=WindowCenter-floor(WindowWidth/2)-RescaleIntercept+1;
%val2=WindowCenter+floor(WindowWidth/2)-RescaleIntercept;
%temp=double(im_in).*double(im_in>val1).*double(im_in<val2)+...
%    val1*double(im_in<=val1)+val2*double(im_in>=val2);
%temp=temp/RescaleSlope;
%im_out=double(temp-val1)/WindowWidth;