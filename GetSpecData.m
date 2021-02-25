%��ȡλ��SpecSamples\�еĹ������ݡ�
function Spec_interp = GetSpecData(WL, filename, SpecNum)
if(~isnumeric(SpecNum))
    SpecNum = 1;
end
WL = WL(:);
Spec_interp = zeros(size(WL, 1), SpecNum);
SpecData = xlsread(['SpecSamples\' filename]);   %�����������
WLdata = SpecData(:,1);
for i = 1:SpecNum
    Spec = SpecData(:,i+1);
    Spec_interp(:,i) = interp1(WLdata, Spec, WL);
end
Spec_interp(isnan(Spec_interp)) = 0;
end