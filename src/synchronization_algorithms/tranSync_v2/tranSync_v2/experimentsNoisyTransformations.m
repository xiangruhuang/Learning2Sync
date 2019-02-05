% Run experiments of denoising noisy pairwise transformations
%
% INPUT: 
% k: number of transformations
% 
% d: dimensionality
%
% sigma: standard deviation of Gaussian noise
%
% transformationType: 'linear', 'affine', 'similarity',
% 'similarityNoReflection', 'euclidean', 'rigid', 'translationOnly'
%
% varyingParameter: string containing the varying parameter
%
% parameterRange: vector containing all values for the varying parameter
% that are to be evaluated
%
% nDraws: number of how many times random transformations are generated
%
% saveImageFolder: output folder for saving .tikz files. Requires the
% matlab2tikz toolbox (if not available, set to [])
%
% varyingParameterLabel: string as label for the varying parameter (to use
% latex syntax)
%
% noiseType: type of noise, currently only additiveGaussian noise is
% supported
%
% nRepetitions: number of times noise is added to the transformations
%
%
% Author: Florian Bernard, f.bernardpi [at] gmail [dot] com
%

function experimentsNoisyTransformations(k, d, sigma, transformationType, ...
    varyingParameter, parameterRange, nDraws, saveImageFolder, ...
    varyingParameterLabel, noiseType, nRepetitions)

%% perform simulation runs for varying parameter

% arrays for storing results
transitiveErrorNoisy = nan(nDraws, numel(parameterRange));
errorNoisy = nan(nDraws, numel(parameterRange));

transitiveErrorDenoised = nan(nDraws, numel(parameterRange));
errorDenoised = nan(nDraws, numel(parameterRange));

for v=1:numel(parameterRange)
    evalc([varyingParameter ' = ' num2str(parameterRange(v))]);
    
    for sampleNo=1:nDraws % repeat experiments
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

        transitiveErrorDenoisedTmp = nan(nRepetitions, 1);
        transitiveErrorNoisyTmp = nan(nRepetitions, 1);
        errorNoisyTmp = nan(nRepetitions, 1);
        errorDenoisedTmp = nan(nRepetitions, 1);
        
		% for the generated set of random transformations, repeatedly add
		% noise
        for r=1:nRepetitions
            switch noiseType
                case 'additiveGaussian'
                    noisyTransCell = addGaussianNoiseToTransformations(...
                        groundTruthTransCell, sigma, transformationType);
                case 'componentWise'
                    noisyTransCell = addNoiseToTransformationsHom(...
                        groundTruthTransCell, sigma, transformationType);
			end
			
            % determine error for noisy transformations
            transitiveErrorNoisyTmp(r) = ...
                transitiveErrorOfPairwiseTransformations(noisyTransCell);
            errorNoisyTmp(r) = ...
                errorTransformations(noisyTransCell, groundTruthTransCell);

            % use transformation synchronisation
            unnoisyTransCell = ...
                synchroniseTransformationsZ(noisyTransCell, transformationType);
           
			% and determine error for synchronised transformations
            transitiveErrorDenoisedTmp(r) = ...
                transitiveErrorOfPairwiseTransformations(unnoisyTransCell);
            errorDenoisedTmp(r) = ...
                errorTransformations(unnoisyTransCell, groundTruthTransCell);
        end
        transitiveErrorNoisy(sampleNo,v) = mean(transitiveErrorNoisyTmp);
        errorNoisy(sampleNo,v) = mean(errorNoisyTmp);
        transitiveErrorDenoised(sampleNo,v) = mean(transitiveErrorDenoisedTmp);
        errorDenoised(sampleNo,v) = mean(errorDenoisedTmp); 
    end
end

%% plot errors
figure('Position', [120 620 300 180]);
switch noiseType
    case 'additiveGaussian'
        titleText = '$\mathcal{\tilde T}^{\mathcal{N}}$';
    case 'componentWise'
        titleText = '$\mathcal{\tilde T}^{C}$';
end
titleText = [titleText ', ' transformationType];
if ( ~strcmp(varyingParameter, 'sigma') )
    titleText = [titleText ', $\sigma{=}' num2str(sigma) '$'];
end
if ( ~strcmp(varyingParameter, 'k') )
    titleText = [titleText ', $k{=}' num2str(k) '$'];
end
if ( ~strcmp(varyingParameter, 'd') )
    titleText = [titleText ', $d{=}' num2str(d) '$'];
end
title([titleText]);

hold on;
plot(parameterRange, mean(errorNoisy,1), 'g', 'LineWidth', 1);
plot(parameterRange, mean(errorDenoised,1), 'b', 'LineWidth', 1);
plot(parameterRange, mean(errorNoisy,1), 'gs', 'MarkerSize', 6, ...
    'MarkerFaceColor', 'g');
plot(parameterRange, mean(errorDenoised,1), 'bs', 'MarkerSize', 6, ...
    'MarkerFaceColor', 'b');

% legend({'noisy', 'synchronised'}, 'Location', 'NorthWest', ...
%     'Interpreter', 'tex');
xlabel(['$' varyingParameterLabel '$']);
ylabel('error');

maxVal = max([mean(errorNoisy,1)  mean(errorDenoised,1)]);
if ( maxVal < 2.5 ) % do not show a finer vertical axis than [0;2.5]
    ylim([0 2.5]);
end

if ( exist('saveImageFolder', 'var') && ~isempty(saveImageFolder) )
    file = [saveImageFolder filesep 'errorPlot-' ...
        transformationType '-k-' num2str(k) ...
       '-d-' num2str(d) '-sigma-' num2str(sigma) ...
       '-noiseType-' noiseType ...
       '-var-' varyingParameter '-nDraws-' num2str(nDraws) '.tikz'];
    
    matlab2tikz(file, ...
        'height', '1cm', 'width', '2cm', ...
        'parseStrings', false);
end
end