startWL = 400;
endWL = 700;
resolution = 1;
TFnum = 16;

WL = startWL:resolution:endWL;

FilteredLEDSpec = GetSpecData(WL, 'Spec_Array.xlsx', TFnum);%calibrated spectra of light sorce array
S_CCD = GetSpecData(WL, 'Spec_VH310G2.xlsx', 1);  % quantom efficiency of the camera VH310G2
k=WL.^-1;
S_CCD=S_CCD./k';
S_CCD = S_CCD(:);%convert the quantom efficiency into power
S_Lens = GetSpecData(WL, 'Spec_Lens.xlsx', 1);%spectrum of the lens
S_Filter = GetSpecData(WL, 'Spec_BPFilter.xlsx', 1);%spectrum of the bandpass filter

filter=FilteredLEDSpec.*S_CCD.*S_Lens.*S_Filter;%the overall spectra
filter = filter/max(max(filter));

load('data/Specs_precise.mat')
trainingDatasize = 200000;
testingDatasize = 100000;
sizeofDataset = trainingDatasize + testingDatasize;
sigma = 0.05;%random noise level
noise = (rand(TFnum, sizeofDataset) - 0.5); %noise
noise = sigma / max(max(abs(noise))) * noise;

tfs_norm=filter'*Specs_norm;

tfs_norm=tfs_norm./max(tfs_norm);%tfs normalization
Specs_norm=Specs_norm./max(tfs_norm);%scaling the spectra and the tfs at the same level

tfs_norm2 = tfs_norm+noise;%add noise
tfs_norm2(tfs_norm2<0) = 0;
maxtfs=max(tfs_norm2);
tfs_norm2=tfs_norm2./maxtfs;
Specs_norm2=Specs_norm./maxtfs;

save('data/Specs_precise_active.mat', 'Specs_norm2', '-v7.3')
save('data/tfs_precise_active.mat', 'tfs_norm2', '-v7.3')