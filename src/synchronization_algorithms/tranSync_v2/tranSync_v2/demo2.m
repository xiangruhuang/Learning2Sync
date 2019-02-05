% setup
d = 3; 
k = 10;
sigma = 0.2;

%% generate ground truth
transformationType = 'linear';

refTransformationsHom = ...
	generateRandomTransformations(d, k, transformationType);

% compute ground truth
groundTruthTransCell = cell(k,k);
for i=1:k
	for j=1:k
		Ti = refTransformationsHom{i};
		Tj = refTransformationsHom{j};
		
		Tij = Ti/Tj;
		
		groundTruthTransCell{i,j} = Tij;
	end
end

%% add gaussian noise
noisyTransCell = addGaussianNoiseToTransformations(...
	groundTruthTransCell, sigma, transformationType);

errorNoisy = ...
	errorTransformations(noisyTransCell, groundTruthTransCell);

disp(['Error between ground truth and noisy transformations is ' num2str(errorNoisy)]);


%% synchronise noisy transformations, i.e. remove noise
unnoisyTransZCell = ...
	synchroniseTransformationsZ(noisyTransCell, transformationType);
	
errorUnnoisyZ = ...
	errorTransformations(unnoisyTransZCell, groundTruthTransCell);

disp(['Error between ground truth and Z-synchronised transformations is ' num2str(errorUnnoisyZ)]);

%% H matrix method
% main method
unnoisyTransHCell = ...
	synchroniseTransformationsH(noisyTransCell, transformationType);

errorUnnoisyH = ...
	errorTransformations(unnoisyTransHCell, groundTruthTransCell);

disp(['Error between ground truth and H-synchronised transformations is ' num2str(errorUnnoisyH)]);

%% visually show matrices

figure('Position',[45 440 1400 360]);
subplot(1,4,1)
imagesc(cell2mat(groundTruthTransCell)), title('ground truth');
colorbar;

subplot(1,4,2)
imagesc(cell2mat(noisyTransCell)), title('noisy');
colorbar;

subplot(1,4,3)
imagesc(cell2mat(unnoisyTransZCell)), title('Z-synchronised');
colorbar;
			
subplot(1,4,4)
imagesc(cell2mat(unnoisyTransHCell)), title('H-synchronised');
colorbar;