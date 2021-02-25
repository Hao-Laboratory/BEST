startWL = 400;
endWL = 700;
WL = 400:1:700;
datasize = 125000;

centerWL_gaus = rand(1, datasize);
Specs_gaus_total = zeros(301, datasize);
for i =1:datasize
    centerWL = centerWL_gaus(1,i)*(endWL-startWL)+startWL;
    Specs_gaus_total(:,i) = gaus(WL,centerWL, 4);
end 

centerWL_gaus = rand(2, datasize);
Specs_gaus_total2 = zeros(301, datasize);
for i =1:datasize
    centerWL1 = centerWL_gaus(1, i)*(endWL-startWL)+startWL;
    centerWL2 = centerWL_gaus(2, i)*(endWL-startWL)+startWL;
    Specs_gaus_total2(:,i) = gaus(WL,centerWL1, 4)+gaus(WL,centerWL2, 4);
end 
Specs_total = Specs_gaus_total;
Specs_total(:, 125001:250000) = Specs_gaus_total2;
Specs_total(:, 250001:500000) = Specs_broad(:, randperm(250000));
Specs_total = Specs_total(:, randperm(500000));
Specs_norm = Specs_total./max(Specs_total);
save('Specs_general.mat','Specs_norm', '-v7.3');