%% general settings
draftMode = true;
% draftMode = false;
saveImageFolder = '';

if ( draftMode )
    nDraws = 2;
else
    nDraws = 100;
end
nRepetitions = 20;

%% experiments with noisy pairwise transformations
% additive gaussian, varying sigma
k = 30;
d = 3;
sigma = nan;
noiseType = 'additiveGaussian';
varyingParameter = 'sigma';
varyingParameterLabel = '\sigma';
parameterRange = 0:0.1:0.5;

% transformationType = 'affine';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
% 
% transformationType = 'linear';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
% 
% transformationType = 'similarity';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
% 
% transformationType = 'euclidean';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)

transformationType = 'euclidean';
experimentsNoisyTransformations(k, d, sigma, transformationType, ...
    varyingParameter, parameterRange, nDraws, saveImageFolder, ...
    varyingParameterLabel, noiseType, nRepetitions)


% % varying k
% k = nan;
% varyingParameterLabel = 'k';
% sigma = 0.5;
% varyingParameter = 'k';
% parameterRange = 10:10:60;
% 
% transformationType = 'affine';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
% 
% transformationType = 'similarity';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
% 
% transformationType = 'linear';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
% 
% transformationType = 'euclidean';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
% 
% transformationType = 'rigid';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
% 
% 
% % varying d
% d = nan;
% k = 30;
% varyingParameterLabel = 'd';
% sigma = 0.2;
% varyingParameter = 'd';
% parameterRange = 2:7;
% 
% transformationType = 'affine';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
% 
% transformationType = 'linear';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
% 
% transformationType = 'similarity';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
% 
% transformationType = 'euclidean';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
% 
% transformationType = 'rigid';
% experimentsNoisyTransformations(k, d, sigma, transformationType, ...
%     varyingParameter, parameterRange, nDraws, saveImageFolder, ...
%     varyingParameterLabel, noiseType, nRepetitions)
