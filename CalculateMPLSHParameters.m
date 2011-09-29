function results = CalculateMPLSHParameters(D, N, ...
    dnnHist, dnnBins, danyHist, danyBins, deltaTarget, r, uHash, uCheck)
% function results = CalculateLSHParameters(D, N, ...
%    dnnHist, dnnBins, danyHist, danyBins, deltaTarget)
%
%%%%%%%%%%%%%%%%  Calculate Optimal LSH Parameters %%%%%%%%%%%%%%%%%%%%%
% Given the nearest neighbor and any statistics, find the optimal LSH 
% parameters.  Based on the ArbitraryDistance OPtimization code.
% 
% Start with the danyHist, danyBins, dnnHist, dnnBins vectors and calculate
% the expected bin populations.
% 
% Input: danyHist, danyBins, dnnHist, dnnBins, N (total number of points)
%    D (number of dimensionsm and error tolerance (prob of missing NN).
%
% Output: optimal parameters for LSH.  Everything is returned in a big
% structure.
% 
% Intermediate Variables (for debugging)
%   dScale - Amount to scale axis to make calculations easier.
%   dnnPDF, danyPDF (xs)
%       Scaled PDF for distances
%   projAnyPDF, projNnPDF (xp)
%       Scaled PDFs for projected differences
%   binAnyProb, binNnProb (wList)
%       Prob of hit as a function of w

% Equation numbers refer to first draft of "Optimal Locality-Sensitive
% Hashing" by Malcolm Slaney, Yury Lifshits and Junfeng He, Submitted to 
% the Proceeings of the IEEE special issue on Web-Scale Multimedia, July
% 2011.

% Copyright (c) 2010-2011 Yahoo!  Inc.   See detailed copyright notice at
% the bottom of this file.

if ~exist('simplepdf','file')
    path(path,'/Users/malcolm/Projects/LSHwithYury/jlab/')
end

debugPlot = 0;          % Set to non zero to get debugging plots

%% 
%%%%%%%%%%%%%%%%%%%   ARGUMENT PARSING    %%%%%%%%%%%%%%%%%%%%%%%%%%

%  Create some fake data for testing
if nargin == 0
    N = 10000;
    D = 4;
end
if nargin < 6
    fprintf('Calculating stats for %d %d-dimensional random points.\n',...
        N, D);
    data = randn(N, D);
    numQueries = min(1000,N);
    nnDistance = zeros(numQueries,1);
    anyDistance = zeros(numQueries,1);
    for i=1:numQueries
        d = sort(sum((repmat(data(i,:),N,1)-data).^2,2));
        nnDistance(i) = d(2);
        anyDistance(i) = d(floor(rand(1,1)*(N-2))+3);
    end
    [dnnHist,dnnBins] = hist(nnDistance, 100);
    [danyHist,danyBins] = hist(anyDistance, 100);
end
if nargin < 7
    deltaTarget = 1/exp(1.0);
end
if nargin < 8
    r = 1;
end
if nargin < 9
    uHash = 1;
end
if nargin < 10
    uCheck = 1;
end

%%%%%%%%%%%%%%%%  Make sure basic data looks good %%%%%%%%%%%%%%%%%%%%%

if debugPlot
    figure(1);
    clf;
    plot(dnnBins, dnnHist/sum(dnnHist), danyBins, danyHist/sum(danyHist));
    legend('Nearest Neighbor', 'Any Neighbor')
    title('Histogram of Distances');
    xlabel('Distance')
    ylabel('Number of Images');
end

% Create the results structure
results = struct('uHash', uHash, 'uCheck', uCheck, 'N', N, 'D', D, ...
    'multiprobeR', r);
results.dnnBins = dnnBins;
results.dnnHist = dnnHist;
results.danyBins = danyBins;
results.danyHist = danyHist;
results.delta = deltaTarget;

%%
%%%%%%%%%%%%%%%%%%%   SCALING    %%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure out scaling before computing PDFs 
% Want to make sure that max dany isn't too big or small so we can invert
% the PDF.
targetMean = 2;
danyMean = sum(danyHist.*danyBins)/sum(danyHist);
dScale = targetMean/danyMean;
fprintf('Scaling all distances by %g for easier probability calcs.\n', ...
    dScale);

%% 
% Create new scales and interpolate everything 
% Create the distance PDFs with the new scale.
% Scale bin sizes so we have enough resolution for calculations
nnBinWidth = min(dnnBins(2:end)-dnnBins(1:end-1))*dScale/2;
xs = 0:nnBinWidth:max(danyBins)*dScale*2;
dnnPDF = max(0, interp1(dnnBins*dScale, dnnHist, xs, 'linear', 0));
danyPDF = max(0, interp1(danyBins*dScale, danyHist, xs, 'linear', 0));
results.xs = xs;
results.dnnPDF = dnnPDF;
results.danyPDF = danyPDF;

if debugPlot
    figure(2);
    clf
    plot(xs/dScale, dnnPDF, xs/dScale, danyPDF);
    legend('Nearest Neighbor', 'Any Neighbor')
    title('Distance Distributions');
    xlabel('Scaled Distance - mean(d_{any}) = 2')
    ylabel('PDF of Distance');
end

%%
%%%%%%%%%%%%%%%%%%%   CREATE AXIS FOR RESULTS    %%%%%%%%%%%%%%%%%%%%%%%%%%
% Be careful to make sure that new distributions fit into bins, no
% truncation.
expansionFactor = 1;       % Amount to increase over distance histograms
maxBin = expansionFactor*max(xs);

% Don't let nnBinWidth get too small because pdfinv creates a square matrix 
nnBinWidth = max(maxBin/1000.0, nnBinWidth);
xp = -maxBin:nnBinWidth:maxBin;
xp(abs(xp)<1e-9) = 1e-9;

results.dScale = dScale;
results.xp = xp;

%%
%%%%%%%%%%%%%%%%%%%   COMPUTE PROJECTION PDFs   %%%%%%%%%%%%%%%%%%%%%%%%%%

% Create the Gaussian N(0,1) distribution we'll use for
% the LSH projection;
lshGaussian = simplepdf(xp, 0, 1, 'gaussian');

% Now multiply the projection PDF with the Gaussian PDF to get the LSH
% projection.
% Don't flip the order.. second term gets inverted and perhaps the
% Gaussian always has enough energy in the tails that we get the
% wrong answer at zero.

projAnyPDF = pdfmult(xp, xs, lshGaussian, danyPDF, xp);
projNnPDF = pdfmult(xp,  xs, lshGaussian, dnnPDF,  xp);
if any(isnan(projNnPDF))
    nnMean = sum(xs .* dnnPDF) / sum(dnnPDF);
    projNnPDF = NormalizePDF(simplepdf(xp, 0, nnMean, 'gaussian'), xp)';
end

results.projAnyPDF = projAnyPDF;
results.projNnPDF = projNnPDF;

if debugPlot
    figure(3);
    subplot(2,1,1);
    vScale = max(dnnPDF*dScale)/max(projNnPDF*dScale);
    plot(xs/dScale, dnnPDF*dScale/vScale, xp/dScale, projNnPDF*dScale); 
    legend('Empirical dnn Distrution', ...
        'Calculated NN LSH Projection', ...
        'Location','NorthWest');
    title('NN LSH Projection');
    xlabel('Distance');
    subplot(2,1,2);
    vScale = max(danyPDF*dScale)/max(projAnyPDF*dScale);
    plot(xs/dScale, danyPDF*dScale/vScale, xp/dScale, projAnyPDF*dScale); 
    legend('Empirical dany Distribution', ...
        'Calculated LSH Projection', ...
        'Location','NorthWest');
    title('Any Projection Histogram');
    xlabel('Distance')
    % pause;
end


%%
%%%%%%%%%%%%%%%%%%%   BUCKET ESTIMATES    %%%%%%%%%%%%%%%%%%%%%%%%%%
% Now compute number of hits vs bin size.

% Create the matrix that convolves the projection data with a triangular
% window to represent the quantization interval.
% wMaxSize = 256*dnnBins(end);
minW = .001;
maxW = 100.0;
numWs = 500;
deltaW = exp(log(maxW/minW)/(numWs-1));
wList = minW * deltaW .^(0:numWs-1);

wMatrix = zeros(length(xp), length(wList));
for i=1:length(wList)
    wMatrix(:,i) = max(0, 1-abs(xp')/wList(i));
end
% Now compute how many hits there are in the center bucket, as a function
% of the bin width.
xpBinWidth = xp(3)-xp(2);      % For example
binAnyProb = projAnyPDF' * wMatrix * xpBinWidth;
binNnProb = projNnPDF' * wMatrix * xpBinWidth;

% Now do the same thing for the multiprobe buckets
wMatrix2 = zeros(length(xp), length(wList));
for i=1:length(wList)
    wMatrix2(:,i) = max(0, 1-abs(xp'-wList(i))/wList(i));
end
binAnyProb2 = projAnyPDF' * wMatrix2 * xpBinWidth;
binNnProb2 = projNnPDF' * wMatrix2 * xpBinWidth;

if 0                % Is this still necessary?
    binAnyProb = min(binAnyProb, 0.9999);
    binAnyProb = max(binAnyProb, 0.0001);
    binNnProb = min(binNnProb, 0.9999);
    binNnProb = max(binNnProb, 0.0001);
    binAnyProb2 = min(binAnyProb2, 0.9999);
    binAnyProb2 = max(binAnyProb2, 0.0001);
    binNnProb2 = min(binNnProb2, 0.9999);
    binNnProb2 = max(binNnProb2, 0.0001);
end


if debugPlot
    figure(4)
    clf;
    semilogx(wList/dScale, [binNnProb' binAnyProb']);
    legend('P_{nn}', 'P_{any}','Location','NorthWest');
    title('LSH Bucket Estimate')
    ylabel('Collision Probabilities')
    xlabel('Bin Width (w)')
    axis([1e-2/dScale 2e1/dScale 0 1]);
    % axis([0 max(wList) 0 1])
    % pause;
end

results.binAnyProb = binAnyProb;
results.binNnProb = binNnProb;
results.binAnyProb2 = binAnyProb2;
results.binNnProb2 = binNnProb2;
results.wList = wList;

%%
%%%%%%%%%%%%%%%%   SIMPLE PARAMETER ESTIMATES    %%%%%%%%%%%%%%%%%%%%%%%

% Now find best LSH parameters using Simple calcs

% results.logNNOverlogAny = log(binNnProb)./log(binAnyProb);
% results.NPower_logNNOverlogAny = N.^(log(binNnProb)./log(binAnyProb));

temp1 = (-log(binAnyProb)/log(N)).^r + ...
    uHash/uCheck * (binAnyProb2./binAnyProb).^r;
costRatioForMultiProbe = (factorial(r))*(binNnProb./binNnProb2).^r .* ...
    temp1;
% results.costRatioForMultiProbe = costRatioForMultiProbe;
results.wSimpleCost =  costRatioForMultiProbe.* ...
    N.^(log(binNnProb)./log(binAnyProb)); 

fixedRatio = results.wSimpleCost;
fixedRatio(isinf(fixedRatio)) = 0;
[simpleMax, simpleBin] = min(fixedRatio);
simpleW = wList(simpleBin)/dScale;
simpleK = floor(-log(N)/log(binAnyProb(simpleBin)));

simpleL = -log(deltaTarget)/ ...
    ( simpleK^r/factorial(r) * (binNnProb(simpleBin)^(simpleK-r)) * ...
        (binAnyProb2(simpleBin)^r) );
simpleL = ceil(simpleL);
% Equations (48), (49) and (50) for simpleBin estimate.
Ch = uHash*(-log(deltaTarget)) * ...
     factorial(r) * (binNnProb(simpleBin)/binNnProb2(simpleBin))^r;
Cc = uCheck * (-log(deltaTarget))*N* ...
    ((binNnProb(simpleBin)./binNnProb2(simpleBin)))^r * ...
    (binAnyProb2(simpleBin)/binAnyProb(simpleBin))^r; 
simpleCost = Ch/( (simpleK^r) * (binNnProb(simpleBin)^simpleK))+ ...
    Cc * (binAnyProb(simpleBin)/binNnProb(simpleBin))^simpleK;

fprintf('Simple Approximation:\n');
fprintf('\tFor %d %d-d data use: ', N, D);
fprintf('w=%g and get %g hits per bin and %g nn.\n', ...
    simpleW, binAnyProb(simpleBin), ...
    binNnProb(simpleBin));
fprintf('\t\tK=%g L=%g cost is %g\n', ...
	simpleK, simpleL, simpleCost);

results.simpleW = simpleW;
results.simpleK = simpleK;
results.simpleL = simpleL;
results.simpleBin = simpleBin;
results.fixedRatio = fixedRatio;
results.simpleCost = simpleCost;

% Now plot the statistics for L=1 for simple parameters.
% For L=1
desiredSimpleK = round(simpleK);
desiredSimpleL = round(simpleL);
nnHitProbL1 = binNnProb(simpleBin)^desiredSimpleK;
anyHitProbL1 = binAnyProb(simpleBin)^desiredSimpleK;

nnHitProb = 1 - (1-nnHitProbL1)^desiredSimpleL;
anyHitProb = 1 - (1-anyHitProbL1)^desiredSimpleL;

fprintf('Expected statistics: for simple approximation\n');
fprintf('\tAssuming K=%d, L=%d, hammingR=%d\n', desiredSimpleK, desiredSimpleL, r);
fprintf('\tProbability of finding NN for L=1: %g\n', nnHitProbL1);
fprintf('\tProbability of finding ANY for L=1: %g\n', anyHitProbL1);
fprintf('\tProbability of finding NN for L=%d: %g\n', desiredSimpleL, nnHitProb);
fprintf('\tProbability of finding ANY for L=%d: %g\n', desiredSimpleL, anyHitProb);
fprintf('\tExpected number of hits per query: %g\n', anyHitProb*N);

%%
%%%%%%%%%%%%%%%%%%%   EXACT PARAMETER ESTIMATION    %%%%%%%%%%%%%%%%%%%%%%
% Now do it with the full optimal calculations
% Eq. 41 for all values of w
alpha =  log((binNnProb-binAnyProb)./(1-binNnProb)) + ...
    r * log(binAnyProb2./binAnyProb) + log(uCheck/uHash) - log(factorial(r));
% Eq. 40 fir all values of w
binK = (log(N)+alpha)./(-log(binAnyProb)) + ...
    r*log((log(N)+alpha)./(-log(binAnyProb)));
binK(imag(binK) ~= 0.0 | binK < 1) = 1;     % Set bad values to at least 1
if 0                % Hopefully not necessary
    for iTemp  = 1:length(binK)
        if isreal(binK(iTemp)) ==0 | binK(iTemp) < 0
            binK(iTemp) = 1;
        end
    end
end

% Now we want to find the total cost for all values of w.  We will argmin
% this for find the best w (This is the inside of Eq. 39.  We first compute
% the inside of x^(log/log)
temp =  ((binK.^r).*(binNnProb-binAnyProb)*N*uCheck.*((binAnyProb2./binAnyProb).^r))./ ...
    ((1-binNnProb).*uHash*factorial(r));
wFullCost = (-log(deltaTarget))* uHash*factorial(r)*((binNnProb./binNnProb2).^r)./ ...
    (binK.^r).*(1-binAnyProb)./(binNnProb-binAnyProb).* ...
        temp.^(log(binNnProb)./log(binAnyProb));
results.wFullCost = wFullCost;

[optimalMin, optimalBin] = min(wFullCost);
optimalW = wList(optimalBin)/dScale;

optimalK = floor(binK(optimalBin));
% optimalL = ceil(-log(deltaTarget)/ ...
%     ( (optimalK^r)/factorial(r) * (binNnProb(optimalBin)^(optimalK-r)) * ...
%       (binNnProb2(optimalBin)^r)));       % Wrong expression for C^r_k -
%       Malcolm 9/8/2011
                                        % Equation (42)
optimalL = ceil(-log(deltaTarget)/ ...
    ( choose(optimalK,r) * (binNnProb(optimalBin)^(optimalK-r)) * ...
      (binNnProb2(optimalBin)^r)));

kVsW = floor(binK);
% Don't let r get bigger than k (vs. W), which can happen for very small w.
binL = ceil(-log(deltaTarget)./ ...
    ( choose(kVsW, min(r, kVsW)) .* (binNnProb.^(binK-r)) .* ...
      (binNnProb2.^r)));

% Equations (48), (49) and (50) for optimalBin estimate.
Ch = uHash * (-log(deltaTarget)) * ...
    factorial(r) * (binNnProb(optimalBin)./ binNnProb2(optimalBin))^r;
Cc = uCheck * (-log(deltaTarget))*N*((binNnProb(optimalBin)./binNnProb2(optimalBin)))^r * ...
    (binAnyProb2(optimalBin)/binAnyProb(optimalBin))^r; 
optimalCost = Ch /( (optimalK^r) * (binNnProb(optimalBin)^optimalK)) + ...
      Cc * (binAnyProb(optimalBin)/binNnProb(optimalBin))^optimalK;

% results.exactCostVsW = wFullCost;
results.exactW = optimalW;
results.exactK = optimalK;
results.exactL = optimalL;
results.exactBin = optimalBin;
results.exactCost = optimalCost;
results.binK = binK;
results.binL = binL;

fprintf('Exact Optimization:\n');
fprintf('\tFor %d %d-d data use: ', N, D);
fprintf('w=%g and get %g hits per bin and %g nn.\n', ...
    optimalW, binAnyProb(optimalBin), ...
    binNnProb(optimalBin));
fprintf('\t\tK=%g L=%g cost is %g\n', ...
	optimalK, optimalL, optimalCost);
 
% And print the statistics for L=1 for simple parameters.
desiredOptimalK = round(optimalK);
desiredOptimalL = round(optimalL);
% nnHitProbL1 = binNnProb(optimalBin)^desiredOptimalK;
% anyHitProbL1 = binAnyProb(optimalBin)^desiredOptimalK;
% From the definition of p_nn in Eq. (46)
nnHitProbL1 = choose(desiredOptimalK, r)*binNnProb(optimalBin)^(desiredOptimalK-r)*...
                            binNnProb2(optimalBin)^(r);
anyHitProbL1 = choose(desiredOptimalK, r)*binAnyProb(optimalBin)^(desiredOptimalK-r)*...
                            binAnyProb2(optimalBin)^(r);
                        
nnHitProb = 1 - (1-nnHitProbL1)^desiredOptimalL;
anyHitProb = 1 - (1-anyHitProbL1)^desiredOptimalL;

fprintf('Expected statistics for optimal solution:\n');
fprintf('\tAssuming K=%d, L=%d, hammingR=%d\n', desiredOptimalK, desiredOptimalL, r);
fprintf('\tp_nn(w) is %g\n', binNnProb(optimalBin));
fprintf('\tp_any(w) is %g\n', binAnyProb(optimalBin));
fprintf('\tProbability of finding NN for L=1: %g\n', nnHitProbL1);
fprintf('\tProbability of finding ANY for L=1: %g\n', anyHitProbL1);
fprintf('\tProbability of finding NN for L=%d: %g\n', desiredOptimalL, nnHitProb);
fprintf('\tProbability of finding ANY for L=%d: %g\n', desiredOptimalL, anyHitProb);
fprintf('\tExpected number of hits per query: %g\n', anyHitProb*N);

%%
if debugPlot
    figure(5);
    clf
    subplot(4,1,1);
    semilogx(wList/dScale, [binNnProb' binAnyProb']);
    legend('Pnn', 'Pany','Location','NorthWest');
    title('LSH Bucket Estimate')
    ylabel('Collision Probabilities')
    xlabel('Bin Width (w)')
    % axis([1e-2/dScale 2e1/dScale 0 1]);

    subplot(4,1,2);
    semilogx(wList/dScale, [log(binNnProb') log(binAnyProb')]);
    title('LSH Bucket Estimate (Log Scale)')
    xlabel('Bin Width (w)');
    ylabel('Log(Collision Probabilities)');
    legend('Log(Pnn)', 'Log(Pany)', 'Location','NorthWest');
    
    subplot(4,1,3);
    semilogx(wList/dScale, wFullCost);
    ylabel('Full Cost')
    xlabel('Bin Width (w)');
    % axis([1e-2/dScale 2e1/dScale 0 5]);
    title('Optimal Cost vs. Bin Width');
    hold on
    semilogx(wList(optimalBin)/dScale, optimalMin, 'rx');
    hold off
    
    subplot(4,1,4);
    semilogx(wList/dScale, fixedRatio);
    ylabel('Cost Exponent')
    xlabel('Bin Width (w)');
    % axis([1e-2/dScale 2e1/dScale 0 5]);
    title('Simplified Cost vs. Bin Width');
    hold on
    semilogx(wList(simpleBin)/dScale, simpleMax, 'rx');
    hold off
end

% save Test6DGaussian N D deltaTarget binNnProb binAnyProb uCheck uHash wList dScale

%%
% Using Equation 17 of Casey's TASLP paper, calculate the underlying
% dimensionality.
%
% Want to compute
%   S = sum x_i where x_i is distance^2
%   L = sum log(x_i) where x_i is distance^2
%   N is the number of points.

% Define
%   Y(x) = log(x) - psi(x)
%   
% Want to compute
%   d = 2 Y^-1(log(S/N) - L/N))

x = danyBins.^2;
S = sum(x .* danyHist);
goodI = danyHist > 0;
L = sum(log(x(goodI)) .* danyHist(goodI));
% N = sum(danyHist);

wantedY = log(S/N)- L/N; 

di = 1:.01:200;
Yi = log(di)-psi(di);
% plot(di,Yi);          % Check the Yi relationship

[ignore,i] = min(abs(wantedY-Yi));
globalD = 2*di(i);

results.globalD = globalD;

%%
% Equation 36 of Casey's paper does the same calculation using the dnn
% information.  We want to calculate
%   d^2 = 4 pi^2/6 mean(x_i)^2 / var(x_i)
% again where x_i is the distance squared.

x = dnnBins.^2;
S = sum(x .* dnnHist);
m = S/sum(dnnHist);

v = sum((x - m).^2 .* dnnHist)/(sum(dnnHist)-1);
localD = sqrt(4*pi^2/6*m^2/v);

results.localD = localD;


% Copyright (c) 2011, Yahoo! Inc.
% All rights reserved.
% 
% Redistribution and use of this software in source and binary forms, 
% with or without modification, are permitted provided that the following 
% conditions are met:
% 
% * Redistributions of source code must retain the above
%   copyright notice, this list of conditions and the
%   following disclaimer.
% 
% * Redistributions in binary form must reproduce the above
%   copyright notice, this list of conditions and the
%   following disclaimer in the documentation and/or other
%   materials provided with the distribution.
% 
% * Neither the name of Yahoo! Inc. nor the names of its
%   contributors may be used to endorse or promote products
%   derived from this software without specific prior
%   written permission of Yahoo! Inc.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
% TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
% PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
% OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
% DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
% THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


function c = choose(n, k)
% function c = choose(n, k)
% Works for vectors of n
c = factorial(n) ./ (factorial(k) .* factorial(n-k));

%%
function normed = NormalizePDF(y,x)

x = [2*x(1)-x(2) x(:)' 2*x(end)-x(end-1)];
dx = (x(3:end)-x(1:end-2))/2;
area = dx .* y;
normed = y/sum(area);


if 0
    t1 = 0:.1:pi;
    y1 = sin(t1);
    t2 = 0:.24:pi;
    y2 = sin(t2);
    n1 = NormalizePDF(y1, t1);
    n2 = NormalizePDF(y2, t2);
    clf; hold on
    plot(t1, n1);
    plot(t2, n2);
    hold off
end

