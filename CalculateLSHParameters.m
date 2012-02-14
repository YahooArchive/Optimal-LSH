function results = CalculateLSHParameters(N, dnnHist, dnnBins, ...
    danyHist, danyBins, deltaTarget, r, uHash, uCheck)
% function results = CalculateLSHParameters(N, dnnHist, dnnBins, ...
%     danyHist, danyBins, deltaTarget, r, uHash, uCheck)
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

% Equation numbers refer to second draft of "Optimal Locality-Sensitive
% Hashing" by Malcolm Slaney, Yury Lifshits and Junfeng He, Submitted to 
% the Proceeings of the IEEE special issue on Web-Scale Multimedia,
% February 2012.

% Copyright (c) 2010-2012 Yahoo!  Inc.   See detailed copyright notice at
% the bottom of this file.

debugPlot = 0;          % Set to non zero to get debugging plots

%% 
%%%%%%%%%%%%%%%%%%%   ARGUMENT PARSING    %%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 5
    fprintf('Syntax: results = CalculateLSHParameters(N, ...\n');
    fprintf('    dnnHist, dnnBins, danyHist, danyBins, ...\n');
    fprintf('    deltaTarget, r, uHash, uCheck);\n');
    return;
end

if ~isscalar(N)
    error('First argument must be a scalar count.');
end
if ~isvector(dnnHist) || ~isvector(dnnHist) || length(dnnHist) ~= length(dnnBins)
    error('dnnHist and dnnBins must be the same length vectors.');
end
if ~isvector(danyHist) || ~isvector(danyHist) || length(danyHist) ~= length(danyBins)
    error('danyHist and danyBins must be the same length vectors.');
end
% Set the default parameter values.
if nargin < 6
    deltaTarget = 1/exp(1.0);
end
if nargin < 7
    r = 1;
end
if nargin < 8
    uHash = 1;
end
if nargin < 9
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
results = struct('uHash', uHash, 'uCheck', uCheck, 'N', N,  ...
    'multiprobeR', r, 'dnnBins', dnnBins, 'dnnHist', dnnHist, ...
	'danyBins', danyBins, 'danyHist', danyHist, 'delta', deltaTarget);

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
% xs is the sampling grid for the distance profiles (positive only)
nnBinWidth = min(dnnBins(2:end)-dnnBins(1:end-1))*dScale/2;
xs = 0:nnBinWidth:max(danyBins)*dScale*2;
dnnPDF = max(0, interp1(dnnBins*dScale, dnnHist, xs, 'linear', 0));
danyPDF = max(0, interp1(danyBins*dScale, danyHist, xs, 'linear', 0));
results.xs = xs;
results.dnnPDF = dnnPDF;
results.danyPDF = danyPDF;
results.dScale = dScale;

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
% xp is the sampling grid for the projection data.
nnBinWidth = max(maxBin/1000.0, nnBinWidth);
xp = -maxBin:nnBinWidth:maxBin;
xp(abs(xp)<1e-9) = 1e-9;

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

deltaMatrix = zeros(length(xp), length(wList));
for i=1:length(wList)
    deltaMatrix(:,i) = max(0, 1-abs(xp')/wList(i));
end
% Now compute how many hits there are in the query's bucket, as a function
% of the bin width.
xpBinWidth = xp(3)-xp(2);      % For example
binAnyProb = projAnyPDF' * deltaMatrix * xpBinWidth;
binNnProb = projNnPDF' * deltaMatrix * xpBinWidth;

% Now do the same thing for the multiprobe buckets
delta2Matrix = zeros(length(xp), length(wList));
for i=1:length(wList)
    % delta2Matrix(:,i) = max(0, 1-abs(xp'-wList(i))/wList(i));
    delta2Matrix(:,i) = max(0, min(1, (1.5 + -2*abs(xp'/wList(i)-.75))));
end
binAnyProb2 = projAnyPDF' * delta2Matrix * xpBinWidth;
binNnProb2 = projNnPDF' * delta2Matrix * xpBinWidth;

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
    semilogx(wList/dScale, [binNnProb' binAnyProb' binNnProb2' binAnyProb2']);
    legend('p_{nn}', 'p_{any}','q_{nn}', 'q_{any}','Location','NorthWest');
    title('LSH Bucket Estimate')
    ylabel('Collision Probabilities')
    xlabel('Bin Width (w)')
    axis([1e-2/dScale 4e1/dScale 0 1]);
end

results.binAnyProb = binAnyProb;
results.binNnProb = binNnProb;
results.binAnyProb2 = binAnyProb2;
results.binNnProb2 = binNnProb2;
results.wList = wList;

%%
%%%%%%%%%%%%%%%%   SIMPLE PARAMETER ESTIMATES    %%%%%%%%%%%%%%%%%%%%%%%

% Now find best LSH parameters using Simple calcs

temp1 = (-log(binAnyProb)/log(N)).^r + ...
    uHash/uCheck * (binAnyProb2./binAnyProb).^r;
costRatioForMultiProbe = (factorial(r))*(binNnProb./binNnProb2).^r .* ...
    temp1;
% results.costRatioForMultiProbe = costRatioForMultiProbe;
fixedRatio = costRatioForMultiProbe .* N.^(log(binNnProb)./log(binAnyProb)); 

% fixedRatio(isinf(fixedRatio)) = nan;
results.wSimpleCost = fixedRatio;
[simpleMax, simpleBin] = min(fixedRatio);
simpleW = wList(simpleBin)/dScale;
simpleK = floor(-log(N)/log(binAnyProb(simpleBin)));

simpleL = -log(deltaTarget)/ ...
    ( simpleK^r/factorial(r) * (binNnProb(simpleBin)^(simpleK-r)) * ...
        (binAnyProb2(simpleBin)^r) );
simpleL = ceil(simpleL);
% Equations (20) and (21) for cost factors
Ch = uHash*(-log(deltaTarget)) * ...
     factorial(r) * (binNnProb(simpleBin)/binNnProb2(simpleBin))^r;
Cc = uCheck * (-log(deltaTarget))*N* ...
    ((binNnProb(simpleBin)./binNnProb2(simpleBin)))^r * ...
    (binAnyProb2(simpleBin)/binAnyProb(simpleBin))^r; 
simpleCost = Ch/( (simpleK^r) * (binNnProb(simpleBin)^simpleK))+ ...
    Cc * (binAnyProb(simpleBin)/binNnProb(simpleBin))^simpleK;

fprintf('Simple Approximation:\n');
fprintf('\tFor %d points of data use: ', N);
fprintf('w=%g and get %g hits per bin and %g nn.\n', ...
    simpleW, binAnyProb(simpleBin), ...
    binNnProb(simpleBin));
fprintf('\t\tK=%g L=%g cost is %g\n', ...
	simpleK, simpleL, simpleCost);

results.simpleW = simpleW;
results.simpleK = simpleK;
results.simpleL = simpleL;
results.simpleBin = simpleBin;
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
% Eq. 30 for all values of w
wAlpha =  log((binNnProb-binAnyProb)./(1-binNnProb)) + ...
    r * log(binAnyProb2./binAnyProb) + log(uCheck/uHash) - log(factorial(r));
% Eq. 32 for all values of w.  Added rprime definition to make things
% easier.
rprime = r ./ (-log(binAnyProb));
wBestK = (log(N)+wAlpha)./(-log(binAnyProb)) + ...
    rprime.*log((log(N)+wAlpha)./(-log(binAnyProb)));
wBestK(imag(wBestK) ~= 0.0 | wBestK < 1) = 1;     % Set bad values to at least 1

% Now we want to find the total cost for all values of w.  We will argmin
% this for find the best w (This is the inside of Eq. 39.  We first compute
% the inside of x^(log/log)
% Equations (20) and (21) for cost factors.
Ch = uHash * (-log(deltaTarget)) * ...
    factorial(r) * (binNnProb./ binNnProb2).^r;
Cc = uCheck * (-log(deltaTarget))*N*((binNnProb./binNnProb2)).^r .* ...
    (binAnyProb2./binAnyProb).^r; 
wFullCost = Ch /( (wBestK.^r) .* (binNnProb.^wBestK)) + ...
      Cc .* (binAnyProb./binNnProb).^wBestK;
results.wExpExactCost = wFullCost;

% Now compute the integer values of k and l.
% Don't let r get bigger than k (vs. W), which can happen for very small w.
kVsW = floor(wBestK);
% We compute L via Equation (17)
lVsW = ceil(-log(deltaTarget)./ ...
    ( choose(kVsW, min(r, kVsW)) .* (binNnProb.^(max(1,wBestK-r))) .* ...
      (binNnProb2.^r)));

[optimalMin, optimalBin] = min(wFullCost);
optimalW = wList(optimalBin)/dScale;

optimalK = kVsW(optimalBin);                             
optimalL = lVsW(optimalBin);

results.exactW = optimalW;
results.exactK = optimalK;
results.exactL = optimalL;
results.exactBin = optimalBin;
results.exactCost = wFullCost(optimalBin);
results.wExactK = wBestK;
results.wExactL = lVsW;
results.wExactCost = wFullCost;

% Now figure out the best possible estimate of the total number of
% candidates we will check.  Sum over all distances < r.
probSum = 0;
for ri = 0:r
    candidatesPerBucket = choose(kVsW, min(ri, kVsW)) .* ...
        results.binAnyProb.^max(0,kVsW-ri) .* ...
        results.binAnyProb2.^ri;
    probSum = probSum + candidatesPerBucket;
end
results.wCandidateCount = N * (1 - (1-probSum).^lVsW);
results.wCandidateCount2 = N*probSum.*lVsW;
results.exactCandidateCount = results.wCandidateCount(optimalBin);

fprintf('Exact Optimization:\n');
fprintf('\tFor %d points of data use: ', N);
fprintf('w=%g and get %g hits per bin and %g nn.\n', ...
    optimalW, binAnyProb(optimalBin), ...
    binNnProb(optimalBin));
fprintf('\t\tK=%g L=%g cost is %g\n', ...
	optimalK, optimalL, wFullCost(optimalBin));
 
% And print the statistics for L=1 for simple parameters.
desiredOptimalK = round(optimalK);
desiredOptimalL = round(optimalL);
% nnHitProbL1 = binNnProb(optimalBin)^desiredOptimalK;
% anyHitProbL1 = binAnyProb(optimalBin)^desiredOptimalK;
% From the definition of p_nn in Eq. (46)
nnHitProbL1 = choose(desiredOptimalK, r) * ...
                            binNnProb(optimalBin)^(desiredOptimalK-r)*...
                            binNnProb2(optimalBin)^(r);
anyHitProbL1 = choose(desiredOptimalK, r) * ...
                            binAnyProb(optimalBin)^(desiredOptimalK-r)*...
                            binAnyProb2(optimalBin)^(r);
                        
nnHitProb = 1 - (1-nnHitProbL1)^desiredOptimalL;
anyHitProb = 1 - (1-anyHitProbL1)^desiredOptimalL;

%%
%%%%%%%%%%%%%%%%%%%   Brute Force Calculation    %%%%%%%%%%%%%%%%%%%%%%%%%%
if 1
    maxK = optimalK + 10;
    Ts = zeros(length(binNnProb), maxK);
    for k=1:maxK
        % Equation 19 for T_s(w)
        cPnnQnn = choose(k, min(r, k))*binNnProb.^(k-r).*binNnProb2.^r;
        cPanyQany = choose(k, min(r, k))*binAnyProb.^(k-r).*binAnyProb2.^r;
        TsByW = uHash * (-log(deltaTarget)) ./ cPnnQnn + ...
            uCheck * N * (-log(deltaTarget)) * cPanyQany ./ cPnnQnn;
        Ts(:,k) = TsByW;
    end
    % http://stackoverflow.com/questions/2635120/how-can-i-find-the-maximum-or-minimum-of-a-multi-dimensional-matrix-in-matlab
    [min_Ts, min_Ts_position] = min(Ts(:));
    % transform the index in the 1D view to 2 indices, given the size of Ts
    [minBruteBin,minBruteK] = ind2sub(size(Ts), min_Ts_position);
    results.Ts = Ts;
    results.bruteBin = minBruteBin;
    results.bruteCost = min_Ts;
    results.bruteK = minBruteK;
    results.bruteW = wList(minBruteBin)/dScale;
    results.bruteCandidates = ...
        N*choose(results.bruteK, min(r, results.bruteK)) * ...
          (binAnyProb(minBruteBin)^(max(1,results.bruteK-r))) * ...
          (binAnyProb2(minBruteBin).^r);
    results.bruteL = ceil(-log(deltaTarget) / ...
        ( choose(results.bruteK, min(r, results.bruteK)) * ...
          (binNnProb(minBruteBin)^(max(1,results.bruteK-r))) * ...
          (binNnProb2(minBruteBin).^r)));
    results.bruteCost = results.bruteL*uHash + results.bruteCandidates*uCheck;

    if debugPlot
        figure(6); 
        clf
        TsDetail = min(results.Ts, 10*min_Ts);
        imagesc(log10(results.wList/results.dScale), ...
            1:size(TsDetail, 2), ...
            log10(TsDetail')); axis ij
        xlabel('Log10 Bin Size (w)'); ylabel('Number of Projections (k)');
        title('Log10(T_s) by Brute Force');
        colorbar;
        hold on; 
        plot(log10(results.bruteW), results.bruteK, 'wo');
        plot(log10(results.exactW), results.exactK,'w*');
        plot(log10(results.simpleW), results.simpleK, 'wx');
        hold off;
    end
end

%%
%%%%%%%%%%%%%%%%%%%   Summarize the results    %%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Expected statistics for optimal solution:\n');
fprintf('\tAssuming K=%d, L=%d, hammingR=%d\n', desiredOptimalK, ...
    desiredOptimalL, r);
fprintf('\tp_nn(w) is %g\n', binNnProb(optimalBin));
fprintf('\tp_any(w) is %g\n', binAnyProb(optimalBin));
fprintf('\tProbability of finding NN for L=1: %g\n', nnHitProbL1);
fprintf('\tProbability of finding ANY for L=1: %g\n', ...
    anyHitProbL1);
fprintf('\tProbability of finding NN for L=%d: %g\n', ...
    desiredOptimalL, nnHitProb);
fprintf('\tProbability of finding ANY for L=%d: %g\n', ...
    desiredOptimalL, anyHitProb);
fprintf('\tExpected number of hits per query: %g\n', ...
    results.wCandidateCount(optimalBin));

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
c = factorial(n) ./ (factorial(k) .* factorial(max(0,n-k)));

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


%%%%%%%%%%%%%%%%%%% JLAB Routines %%%%%%%%%%%%%%%%%%%%%%%%%
% The following routines were written by J. M. Lilly
% as part of the Matlab jlab package.  The entire package can be found at
%           http://www.jmlilly.net/jmlsoft.html
% I am grateful for permission to include Dr. Lilly's PDF code in this
% package.
%%%%%%%%%%%%%%%%%%% JLAB Routines %%%%%%%%%%%%%%%%%%%%%%%%%

%JLAB_LICENSE  License statement and permissions for JLAB package.
%
%   Copyright (C) 1993--2011 J.M. Lilly
%   _______________________________________________________________________
%
%   Citation
%
%   If you use this software in research resulting in a scientific
%   publication, the software should be acknowledged and cited as
%
%   Lilly, J. M. (2011), JLAB: Matlab freeware for data analysis, 
%       Version 0.92, http://www.jmlilly.net/software.html.
%   _______________________________________________________________________
%
%   License
%
%   JLAB is distributed under the 
%   
%     "Creative Commons Attribution Noncommercial-Share Alike License"
%
%   Version 3.0, available at 
%
%       http://creativecommons.org/licenses/by-nc-sa/3.0/us/
%
%   You are free:
%
%       To Share -- To copy, distribute and transmit the work
%
%       To Remix -- To adapt the work
%
%   Under the following conditions:
%
%       Attribution -- You must attribute the work in the manner specified
%          by the author or licensor (but not in any way that suggests that 
%          they endorse you or your use of the work).
%
%       Noncommercial -- You may not use this work for commercial purposes.
%
%       Share Alike -- If you alter, transform, or build upon this work, 
%          you may distribute the resulting work only under the same or 
%          similar license to this one.
%
%   See the above link for the full text of the license.
%   _______________________________________________________________________
%
%   Disclaimer
%
%   This software is provided 'as-is', without any express or implied
%   warranty. In no event will the author be held liable for any damages
%   arising from the use of this software.
%   _______________________________________________________________________
%
%   This is part of JLAB --- type 'help jlab' for more information 
%   (C) 1993--2011 J.M. Lilly --- type 'help jlab_license' for details

function[f]=simplepdf(x,mu,sig,flag)
%SIMPLEPDF  Gaussian, uniform, Cauchy, and exponential pdfs.
%
%   F=SIMPLEPDF(X,MU,SIG,'gaussian') computes a Gaussian pdf with mean
%   MU and standard deviation SIG.
%  
%   F=SIMPLEPDF(X,MU,SIG,'boxcar') computes a uniform pdf with mean MU
%   and standard deviation SIG.
%  
%   F=SIMPLEPDF(X,XO,wAlpha,'cauchy') computes a Cauchy pdf with location
%   parameter XO and scale parameter wAlpha.
%
%   F=SIMPLEPDF(X,BETA,'exponential') computes an exponential pdf with
%   scale parameter, hence mean and standard deviation, equal to BETA.
%
%   'simplepdf --f' generates a sample figure
%
%   Usage: f=simplepdf(x,mu,sig,'gaussian');
%          f=simplepdf(x,mu,sig,'boxcar');
%          f=simplepdf(x,xo,wAlpha,'cauchy');
%          f=simplepdf(x,beta,'exponential');
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2001--2008 J.M. Lilly --- type 'help jlab_license' for details    
  
warning('off','MATLAB:divideByZero')
  
if strcmp(x,'--f')
  simplepdf_fig;
  return
end
dx=x(2)-x(1);

if nargin==3
    flag=sig;
end

if nargin<3&&strcmp(flag,'exponential')||nargin<4&&~strcmp(flag,'exponential')
    error('Not enough input arguments.')
end

if strcmp(flag,'gaussian')
  f=exp(-(x-mu).^2./2./sig.^2)./sig./sqrt(2*pi);
elseif strcmp(flag,'boxcar')
  f=0*x;
  ia=min(find(x-mu>-3.4641*sig/2))-1;
  ib=min(find(x-mu>3.4641*sig/2));
  f(ia:ib)=1;
  f=f./vsum(f*dx,1);
elseif strcmp(flag,'cauchy')
  wAlpha=sig;
  f=frac(wAlpha./pi,(x-mu).^2 + wAlpha.^2);
elseif strcmp(flag,'exponential')
  f=frac(1,mu).*exp(-abs(x)./mu);
end


warning('on','MATLAB:divideByZero')

function[]=simplepdf_fig

x=(-100:.1:100)';
mu=25;
sig=10;
f=simplepdf(x,mu,sig,'gaussian');
%[mu2,sig2]=pdfprops(x,f);
figure,plot(x,f),vlines(mu,'r')
%a=conflimit(x,f,95);
%vlines(mu+a,'g'),vlines(mu-a,'g')
title('Gaussian with mean 25 and standard deviation 10')
function[fz]=pdfmult(xi,yi,fx,fy,zi)
%PDFMULT  Probability distribution from multiplying two random variables.
%
%   FZ=PDFMULT(XI,YI,FX,FY,ZI), given two probability distribution
%   functions FX and FY defined over XI and YI, returns the pdf FZ
%   corresponding to Z=X*Y over values ZI.
%
%   PDFMULT uses PDFDIVIDE.
%
%   Usage: yn=pdfmult(xi,yi,fx,fy,zi);
%
%   'pdfmult --f' generates a sample figure.  
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2001--2009 J.M. Lilly --- type 'help jlab_license' for details    
  
if strcmp(xi,'--f')
    pdfmult_fig;
    return
end

vcolon(xi,yi,fx,fy,zi);
fyi=jinterp(yi,fy,zi);
fxi=jinterp(xi,fx,zi);
vswap(fyi,nan,0);
vswap(fxi,nan,0);
fyinv=pdfinv(zi,fyi);
fz=pdfdivide(zi,zi,fxi,fyinv,zi);
[mu,sigma]=pdfprops(zi,fz);

function[]=pdfmult_fig

dx=0.1;
dy=0.05;
dz=0.025;
s1=1;
s2=2;
xi=(-10:dx:10)';
yi=(-10:dy:10)';
zi=(-10:dz:10)';

fx=simplepdf(xi,0,s1,'gaussian');
fy=simplepdf(yi,0,s2,'gaussian');

fz0=s1.*s2./pi./(s2.^2.*xi.^2+s1.^2);
fz0=fz0./vsum(fz0*dx,1);
fz=pdfmult(xi,yi,fx,fy,zi);

figure,plot(zi,fz),hold on,plot(xi,fx)
plot(yi,fy)
linestyle default
title('RV with green pdf multiplied by RV with red pdf equals RV with blue pdf')
x1=randn(100000,1)*s1;
y1=randn(100000,1)*s2;
[fz1,n]=hist(x1.*y1,(-10:.1:10));
plot(n,fz1/10000,'.')

text(4,0.4,'Green and red are Gaussian')
text(4,0.35,'Blue is disribution of product')
text(4,0.30,'Dots are from a random trial')

axis([-10 10 0 0.45])
function[varargout]=vcolon(varargin)
%VCOLON  Condenses its arguments, like X(:).
%
%   [Y1,Y2, ... YN]=VCOLON(X1,X2, ... XN) is equivalent to   
%		
%      Y1=X1(:); Y2=X2(:); ... YN=XN(:);    
%
%   VCOLON(X1,X2,...XN) with no output arguments overwrites the 
%   original input variables.
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2000, 2004 J.M. Lilly --- type 'help jlab_license' for details    
    
if strcmp(varargin{1}, '--t')
   vcolon_test,return
end
 
for i=1:nargin
  x=varargin{i};
  varargout{i}=x(:);
end

eval(to_overwrite(nargin));


function[]=vcolon_test
x=[1 3; 2 4];
y=x;

vcolon(x,y);
reporttest('VCOLON ', all(x==(1:4)'&y==(1:4)'))
function[]=reporttest(str,bool)
%REPORTTEST  Reports the result of an m-file function auto-test.
%
%   Called by JLAB_RUNTESTS.
%   _________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2003--2009 J.M. Lilly --- type 'help jlab_license' for details


global BOOL_JLAB_RUNTEST
global FUNCTION_NUMBER
global FUNCTION_NUMBER_ARRAY

FUNCTION_NUMBER_ARRAY=[FUNCTION_NUMBER_ARRAY;FUNCTION_NUMBER];

if bool
    disp([str ' test: passed'])
    BOOL_JLAB_RUNTEST=[BOOL_JLAB_RUNTEST;1];
else  
    disp([str ' test: FAILED'])
    BOOL_JLAB_RUNTEST=[BOOL_JLAB_RUNTEST;0];
end
function[str]=to_overwrite(N)
%TO_OVERWRITE Returns a string to overwrite original arguments.
%
%   STR=TO_OVERWRITE(N), when called from within an m-file which has
%   VARARGIN for the input variable, returns a string which upon
%   EVAL(STR) will cause the first N input variables in the caller
%   workspace with the values contained in the first N elements of
%   VARARGOUT.
%
%   See also TO_GRAB_FROM_CALLER.
%
%   Usage: eval(to_overwrite(N))
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2001--2006 J.M. Lilly --- type 'help jlab_license' for details      

str{1}=     'if nargout==0';
str{end+1}= '   global ZZoutput';
str{end+1}= '   evalin(''caller'',[''global ZZoutput''])';
str{end+1}=['   for i=1:' int2str(N)];
str{end+1}= '     if ~isempty(inputname(i))';
str{end+1}= '       ZZoutput=varargout{i};';
str{end+1}= '       assignin(''caller'',inputname(i), ZZoutput)';
str{end+1}= '     end';
str{end+1}= '   end';
str{end+1}='   evalin(''caller'',[''clear ZZoutput''])';
str{end+1}='end';

str=strs2row(str);


function[row]=strs2row(x)
%STRS2ROW  Converts a cell array of strings into a row array

M=length(x);
for i=1:M
    n(i)=length(x{i});
end
N=max(n);

row=[];

for i=1:M
    row=[row,char(10),x{i}]; 
end
function[yi]=jinterp(x,y,xi,str)
%JINTERP Matrix-matrix 1-D interpolation.
%
%   YI=JINTERP(X,Y,XI), returns the linear interpolation of Y onto XI     
%   based on the functional relationship Y(X).                            
%                                                                         
%   Unlike INTERP1, JINTERP allows X,Y, and XI to be either vectors or    
%   matrices. If more than one argument is a matrix, those matrices must  
%   be of the same size. YI is a matrix if any input argument is a        
%   matrix. All vectors should be column vectors and all matrices should  
%   have data in columns.                                                 
%                                                                         
%   Also, only data points of XI in between the maximum and minimum       
%   values of X are interpolated.                                         
%                                                                         
%   This useful, for example, in interpolating section data with          
%   nonuniform pressure levels onto standard levels.                      
%                                                                         
%   See also INTERP1.                                                      
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2000--2008 J.M. Lilly --- type 'help jlab_license' for details    
if nargin~=4
   str='linear';
end

%convert row vectors to column vectors
if size(x,1)==1
   x=conj(x');
end
if size(y,1)==1
   y=conj(y');
end
if size(xi,1)==1
   xi=conj(xi');
end

%ensure sizes are compatible
Lx=size(x,2);
Ly=size(y,2);
Lxi=size(xi,2);
maxL=max([Lx Ly Lxi]);
bool=(Lx==1|Lx==maxL)&(Ly==1|Ly==maxL)&(Lxi==1|Lxi==maxL);
if ~bool,
   error('Arguments are not of compatible size')
end

%convert vectors to matrices
if Lx==1
   x=x*ones(1,maxL);
end
if Ly==1
   y=y*ones(1,maxL);
end
if Lxi==1
   xi=xi*ones(1,maxL);
end

yi=nan*ones(size(xi,1),maxL);

%check x for monotonicity
mdx=min(min(diff(x)));
if mdx<=0
   disp('Ensuring monotonicity of X by adding noise and sorting.')
   mx=min(min(x));
   x=x+randn(size(x))/1000/1000;
   x=sort(x,1);
end

for i=1:size(x,2)
 	colmin=min(x(isfinite(x(:,i)),i));
	colmax=max(x(isfinite(x(:,i)),i));

%	a=min(find(xi(:,i)>=colmin));
%	b=max(find(xi(:,i)<=colmax));
	index=find(xi(:,i)>=colmin&xi(:,i)<=colmax&isfinite(xi(:,i)));
	
	if ~isempty(index)>=0,
		yi(index,i)=interp1(x(:,i),y(:,i),xi(index,i),str);
	end
end




function[b]=allall(x)
%ALLALL(X)=ALL(X(:))
%   _________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information 
%   (C) 2004 J.M. Lilly --- type 'help jlab_license' for details
b=all(x(:));
function[fz]=pdfinv(yi,fy)
%PDFINV  Probability distribution of the inverse of a random variable.
%
%   YN=PDFMULT(YI,FY) given a probability distribution functions FY
%   defined over YI, returns the pdf of the inverse random variable 1/Y.
%
%   'pdfinv --t' runs a test
%   'pdfinv --f' generates a sample figure
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information (C)
%   2001, 2004 J.M. Lilly --- type 'help jlab_license' for details
  
if strcmp(yi,'--t')
  pdfinv_test;
  return
end

if strcmp(yi,'--f')
  pdfinv_fig;
  return
end

%tol=1e-10;
%index=find(yi==0);
%if ~isempty(index)
%  yi(index)=1e-10;
%end

index=1:round(length(yi)/2)-1;
N=round(length(yi)/2)-1;

%index=(N:-1:1  length(yi):-1:N+1);
%fz=pdfchain(yi(index),fy(index),1./yi(index),yi);
warning('off','MATLAB:divideByZero')
fz=pdfchain(yi,fy,1./yi,yi);
warning('on','MATLAB:divideByZero')

vswap(fz,nan,0);

function[]=pdfinv_test
  
wAlpha=2;

dy=0.01;
yi=(-40:dy:40)';
fx=simplepdf(yi,0,wAlpha,'cauchy');
fy=simplepdf(yi,0,1./wAlpha,'cauchy');
fy2=pdfinv(yi,fx);

tol=1e-3;
bool=vmean(abs(fy-fy2).^2,1)<tol;
reporttest('PDFINV for Cauchy, Papoulis special case p. 94',bool)

function[]=pdfinv_fig

s2=2;
dy=0.01;
yi=(-40:dy:40)';
fy=simplepdf(yi,0,s2,'gaussian');

fz=pdfinv(yi,fy);

y1=randn(100000,1)*s2;
[fz1,n]=hist(1./y1,(-11:.1:11));

figure,
plot(yi,fz)
hold on
plot(n,fz1/10000,'.'),xlim([-10 10])
title('PDF of the inverse of a Gaussian RV')
text(4,0.30,'Dots are from a random trial')
function[fy]=pdfchain(x,fx,g,yi)
%PDFCHAIN  The "chain rule" for probabilty density functions.
%
%   FY=PDFCHAIN(X,FX,G,Y), where FX is a probability density function
%   defined over values X, and G is some function of X, returns the
%   probility density function of random variable G(X) over values Y.  
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2001, 2004 J.M. Lilly --- type 'help jlab_license' for details    
  
warning('off','MATLAB:divideByZero')

vcolon(g,x,fx);

index=find(~isnan(x)&~isnan(fx)&~isnan(g));

x=x(index);
fx=fx(index);
g=g(index);

[g,index]=sort(g);
vindex(x,fx,index,1);

gprime=vdiff(g,1);
%gprime=gprime*ones(size(fx(1,:)));
%x=x*ones(size(fx(1,:)));
%g=g*ones(size(fx(1,:)));

fyx=fx./abs(gprime);
fyx(1)=0;
fyx(end)=0;
index=find(~isfinite(fyx)|fyx==0);
index=index(2:end-1);
if ~isempty(index)
   fyx(index)=fyx(index-1)./2+fyx(index+1)./2;
end
%figure,plot(fyx)
fy=jinterp(g,fyx,yi);

dy=yi(2)-yi(1);
fy=fy./(ones(size(fy(:,1)))*vsum(fy*dy,1));

warning('on','MATLAB:divideByZero')

function[varargout]=vindex(varargin)
%VINDEX  Indexes an N-D array along a specified dimension.
%
%   Y=VINDEX(X,INDEX,DIM) indexes the multidimensional array X along     
%   dimension DIM. This is equivalent to   
%		
%		    1 2       DIM     DIMS(X)
%		    | |        |         |
%		Y=X(:,:, ... INDEX, ..., :);		
%
%   where the location of INDEX is specified by DIM.
%
%   VINDEX is defined to return an empty array if INDEX is empty.
%  
%   Note that VINDEX does not index along singleton dimensions, thus
%   when X is a column vector, VINDEX(X,INDEX,2) returns X.  
%
%   [Y1,Y2,...YN]=VINDEX(X1,X2,...XN,INDEX,DIM) also works.
%
%   VINDEX(X1,X2,...XN,INDEX,DIM); with no output arguments overwrites 
%   the original input variables.
%
%   VINDEX also supports logical indexing with INDEX a boolean array
%   of the same size as the dimension being indexed.
%
%   See also VINDEXINTO, SQUEEZE, DIMS, PERMUTE, SHIFTDIM.
%
%   'vindex --t' runs a test.
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2001--2009 J.M. Lilly --- type 'help jlab_license' for details    


if strcmp(varargin{1}, '--t')
  vindex_test,return
end

index=varargin{end-1};
dim=varargin{end};

nvars=nargin-2;
vars=varargin(1:nvars);

%eval(to_grab_from_caller(2))  %assigns vars, varnames, nvars

for i=1:nvars
  varargout{i}=vindex1(vars{i},index,dim);
end

eval(to_overwrite(nargin-2));

%now to_overwrite uses
%varnames not inputnames ... and does not take input argument
   
function[y]=vindex1(x,index,dim)  

y=[];
if ~isempty(x)
    if size(x,dim)==1
        %X has only one element along dimension DIM; do nothing
        if ~isempty(index)
            y=x;
        end
    else
        %You would think Matlab would provide a simpler way to do this.
        str='y=x(';
        ndx=length(find(size(x)>=1));
        if ~isempty(index)
          for i=1:ndx
              if i~=dim
                  str=[str ':,'];
              else
            str=[str 'index,'];
              end
          end              
          str=[str(1:end-1) ');'];
          eval(str);
        end
    end
end

function[]=vindex_test
x1=[1 1; 2 2];
index=2;
ans1=x1(:,index);
vindex(x1,index,2);
reporttest('VINDEX col case', aresame(x1,ans1))

x1=[1 1; 2 2];
index=2;
ans1=x1(index,:);
vindex(x1,index,1);
reporttest('VINDEX row case', aresame(x1,ans1))

x1=[1 1; 2 2];
index=[false; true];
ans1=x1(:,index);
vindex(x1,index,2);
reporttest('VINDEX col case, logical', aresame(x1,ans1))

x1=[1 1; 2 2];
index=[false; true];
ans1=x1(index,:);
vindex(x1,index,1);
reporttest('VINDEX row case, logical ', aresame(x1,ans1))

x1=[1 2];
index=1:10;
ans1=x1;
vindex(x1,index,1);
reporttest('VINDEX row vector indexed along rows case', aresame(x1,ans1))

x1=[1 2]';
index=1:10;
ans1=x1;
vindex(x1,index,2);
reporttest('VINDEX column vector indexed along columns case', aresame(x1,ans1))

x1=[1 2]';
index=[];
ans1=[];
vindex(x1,index,2);
reporttest('VINDEX empty index case', aresame(x1,ans1))

x1=[];
index=(1:2);
ans1=[];
vindex(x1,index,2);
reporttest('VINDEX empty array case', aresame(x1,ans1))
function[varargout]=vdiff(varargin)
%VDIFF	Length-preserving first central difference.
%
%   DX=VDIFF(X,DIM) differentiates X along dimension DIM using the first 
%   central difference; DX is the same size as X.                                 
%                                                                        
%   [D1,D2,...,DN]=VDIFF(X1,X2,...,XN,DIM) for multiple input variables 
%   also works. 
%
%   VDIFF(X1,X2,...,DIM); with no output arguments overwrites the
%   original input variables.
%
%   DXDT=VDIFF(DT,...) optionally uses scalar timestep DT to approximate
%   a time derivative, i.e. DXDT equals DX divided by DT.
%   _____________________________________________________________________
%
%   First and last points
%
%   The first and last points must be treated differently, as the central 
%   difference is not defined there.  Three different methods can be used.
%
%   VDIFF(...,STR) specifies which method to use.
%
%        'endpoint'  uses the first forwards / first backwards difference
%                    at the first and last point, respectively.  
%        'periodic'  treats the array as being periodic along dimension DIM,
%                    so that the central difference is defined at endpoints.
%        'nans'      fills in the first and last values with NANs.
%
%   The default behavior is 'endpoint'.
%   _____________________________________________________________________
%
%   'vdiff --t' runs some tests.
%
%   Usage:  x=vdiff(x,dim);
%           x=vdiff(dt,x,dim);
%           x=vdiff(dt,x,dim,'periodic');
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2000--2011 J.M. Lilly --- type 'help jlab_license' for details    
  
%   I am so irritated by diff

 
if strcmp(varargin{1}, '--t')
  vdiff_test,return
end


if length(varargin{1})==1
    dt=varargin{1};
    varargin=varargin(2:end);
else
    dt=1;
end

if ischar(varargin{end})
    str=varargin{end};
    varargin=varargin(1:end-1);
else
    str='endpoints';
end
    
%if length(varargin{end})==1
   n=varargin{end};
   varargin=varargin(1:end-1);
%else
 %  n=1;
%end

for i=1:length(varargin)
  varargout{i}=vdiff1(varargin{i},n,str)./dt;
end

if nargin>1
  eval(to_overwrite(length(varargin)))
end

function[y]=vdiff1(x,n,str)
  	  
if ~isempty(x)
    y=vshift(x,1,n)./2-vshift(x,-1,n)./2;
    %y=vshift(x,1,n)-x;
        
    if strcmp(str(1:3),'end')
        y=vindexinto(y,vindex(x,2,n)-vindex(x,1,n),1,n);
        y=vindexinto(y,vindex(x,size(x,n),n)-vindex(x,size(x,n)-1,n),size(x,n),n);
        %y(1,:)=x(2,:)-x(1,:);
        %y(end,:)=x(end,:)-x(end-1,:);
    elseif strcmp(str(1:3),'nan')
        y=vnan(y,1,n);
        y=vnan(y,size(y,n),n);
    elseif strcmp(str(1:3),'per')
        %Do nothing
    end
else
    y=[];
end
    
function[]=vdiff_test

y1=(1:4)';
y2=2*(1:4)';
[x1,x2]=vdiff(y1,y2,1);
bool=aresame(x1,[1 1 1 1]').*aresame(x2,2*[1 1 1 1]');
reporttest('VDIFF', bool)
vdiff(y1,y2,1);
bool=aresame(y1,[1 1 1 1]').*aresame(y2,2*[1 1 1 1]');
reporttest('VDIFF output overwrite', bool)

dt=pi;
y1=(1:4)';
y2=2*(1:4)';
[x1,x2]=vdiff(pi,y1,y2,1);
bool=aresame(x1,[1 1 1 1]'./dt).*aresame(x2,2*[1 1 1 1]'./dt,1e-10);
reporttest('VDIFF with non-unit time step', bool)

function[varargout]=vshift(varargin)
% VSHIFT  Cycles the elements of an array along a specified dimension.
%
%   Y=VSHIFT(X,N,DIM) cycles the elements of X N places along dimension DIM.
%  
%   Example: x=[1 2 3 4 5];
%            vshift(x,+1,2)=[2 3 4 5 1]           
%            vshift(x,-1,2)=[5 1 2 3 4]           
%
%   Note shifting by N and then by -N recovers the original array. 
%
%   [Y1,Y2,...YN]=VSHIFT(X1,X2,...XN,N,DIM) also works.
%
%   VSHIFT(X1,X2,...XN,N,DIM); with no arguments overwrite the original 
%   input variables.
%
%   ------------------------------------------------------------------
%   Y=VSHIFT(X,N,DIM,INDEX,DIM2) applies this shift selectively, only to
%   that subset of X obtained by indexing X with INDEX along DIM2, i.e.
%
%		    1 2      DIM2     DIMS(X)
%		    | |        |         |
%		  X(:,:, ... INDEX, ..., :)	
%
%   is cycled N places along dimension DIM, but the remainder of X is not. 
%   DIM and DIM2 cannot be the same.  The above extensions to multiple 
%   output varibles work in this case as well.  
%   ------------------------------------------------------------------
%
%   See also: VINDEX  
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2001--2006 J.M. Lilly --- type 'help jlab_license' for details    
  

if strcmp(varargin{1}, '--t')
  vshift_test,return
end

%/********************************************************
%Sort out input arguments
nax=2;
if nargin>4
  if  length(varargin{end-3}(:))==1
     nax=4;
     dim2=(varargin{end});
     jj=varargin{end-1}(:);
     dim=varargin{end-2};
     n=varargin{end-3};
     if dim==dim2
       error('DIM and DIM2 cannot be the same.')
     end
     %n, jj, dim,dim2
  end
end

if nax==2
  dim=varargin{end};
  n=varargin{end-1};
end
%\********************************************************

for i=1:length(varargin)-nax
  if nax==2
    varargout{i}=vshift1(varargin{i},n,dim);
  else
    varargout{i}=vshift1(varargin{i},n,dim,jj,dim2);
  end  
end
eval(to_overwrite(nargin-nax))


function[y]=vshift1(x,n,ndim,jj,ndim2)

N=size(x,ndim);
if n>0
   ii=[(n+1:N) (1:n)];
elseif n<0
   n=-n;
   ii=[(N-(n-1):N) (1:N-n)];
elseif n==0
   ii=(1:N);
end

if nargin==3
    y=vindex(x,ii,ndim);
else
    y1=vindex(x,jj,ndim2);
    y1=vindex(y1,ii,ndim);
    %vsize(x,y1,jj,ndim2);
    %x,y1,ii,jj,ndim2
    y=vindexinto(x,y1,jj,ndim2);   
end


function[]=vshift_test
x=(1:10);
ans1=[(2:10) 1];
reporttest('VSHIFT col case', aresame(vshift(x,1,2),ans1))

x=(1:10)';
ans1=[10 (1:9)]';
reporttest('VSHIFT row case', aresame(vshift(x,-1,1),ans1))

clear x ans1
x(:,:,1)=[1 2; 3 4];
x(:,:,2)=2*[1 2; 3 4];
ans1(:,:,2)=x(:,:,1);
ans1(:,:,1)=x(:,:,2);
reporttest('VSHIFT mat case', aresame(vshift(x,1,3),ans1))

clear x ans1
x(:,:,1)=[1 2; 3 4];
x(:,:,2)=2*[1 2; 3 4];
ans1(:,:,1)=[3 2;1 4];
ans1(:,:,2)=2*[3 2;1 4];
reporttest('VSHIFT mat selective case one', aresame(vshift(x,1,1,1,2),ans1))

clear x ans1
x(:,:,1)=[1 2; 3 4];
x(:,:,2)=2*[1 2; 3 4];
ans1(:,:,1)=[3 4;1 2];
ans1(:,:,2)=2*[3 4;1 2];
reporttest('VSHIFT mat selective case two', aresame(vshift(x,1,1,1:2,2),ans1))

function[varargout]=vindexinto(varargin)
%VINDEXINTO  Indexes into N-D array along a specified dimension.
%
%   Y=VINDEXINTO(Y,X,INDEX,DIM) indexes the array X into the multi- 
%   dimensional array Y along dimension DIM. This is equivalent to   
%		
%		    1 2       DIM     DIMS(X)
%		    | |        |         |
%		  Y(:,:, ... INDEX, ..., :)=X;		
%
%   where the location of INDEX is specified by DIM.  X must have the
%   exact same size as the block it replaces, or be a scalar.  
%   
%   VINDEXINTO is defined leave Y unchanged if INDEX is empty, and to 
%   ignore NANs and INFs in INDEX.  
%
%   VINDEXINTO(Y,X,INDEX,0) with DIM=0 is equivalent to Y(INDEX)=X;
%  
%   [Y1,Y2,...YN]=VINDEXINTO(Y1,Y2,...YN,X1,X2,...XN,INDEX,DIM) also 
%   works.
%
%   VINDEXINTO(Y1,Y2,...YN,X1,X2,...XN,INDEX,DIM); with no output 
%   arguments overwrites the original input variables ,Y2,...YN.
%
%   See also VINDEX, SQUEEZE, DIMS, PERMUTE, SHIFTDIM
%
%   'vindexinto --t' runs a test.
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2005--2006 J.M. Lilly --- type 'help jlab_license' for details    
  
if strcmp(varargin{1}, '--t')
  vindexinto_test,return
end

index=varargin{end-1};
dim=varargin{end};

nvars=nargin-2;
if ~iseven(nvars)
  error('There should be an even number of input arguments.')
end

vars=varargin(1:nvars/2);
fromvars=varargin(nvars/2+1:nvars);

for i=1:nvars/2
varargout{i}=vindexinto1(vars{i},fromvars{i},index,dim);
end
eval(to_overwrite(nvars/2));

function[y]=vindexinto1(y,x,index,dim)  
%You would think Matlab would provide a simpler way to do this.

if dim==0
    y(index)=x;
else
    str='y(';

    %[index,sorter]=sort(index);
    %x=x(sorter);

    ii=find(isfinite(index));
    if ~isempty(ii)
      index=index(ii);
    else
      x=[];
      index=[];
    end


    ndx=length(find(size(y)>=1));
    if ~isempty(index) && ~isempty(x)
      for i=1:ndx
          if i~=dim
              str=[str ':,'];
          else
        str=[str 'index,'];
          end
      end
      str=[str(1:end-1) ')=x;'];
      eval(str);
    end
end

if any(~isfinite(y(:)))
  if all(isreal(y(isfinite(y))))
      y(~isfinite(y(:)))=nan;
  else
      y(~isfinite(y(:)))=nan+sqrt(-1)*nan;
  end
end      
    

function[]=vindexinto_test
y1=[1 1; 2 2];
index=2;
x=[5 6]';
ans1=y1;
ans1(:,2)=x;
vindexinto(y1,x,index,2);
reporttest('VINDEXINTO col case', aresame(y1,ans1))

y1=[1 1; 2 2];
index=2;
x=[5 6];
ans1=y1;
ans1(2,:)=x;
vindexinto(y1,x,index,1);
reporttest('VINDEXINTO row case', aresame(y1,ans1))

clear y1 ans1
y1(:,:,1)=[1 2; 3 4];
y1(:,:,2)=2*[1 2; 3 4];
index=1;
x=[5 6]';x=vrep(x,2,3);
ans1(:,:,1)=[5 6; 3 4];
ans1(:,:,2)=[5 6; 2*3 2*4];
vindexinto(y1,x,index,1);
reporttest('VINDEXINTO 3-D col case', aresame(y1,ans1))

vindexinto(y1,[],index,1);
reporttest('VINDEXINTO empty x case', aresame(y1,ans1))


y1=[1 1; 2 2];
index=[1 4];
x=[3 5];
ans1=[3 1; 2 5];
vindexinto(y1,x,index,0);
reporttest('VINDEXINTO DIM=0 case', aresame(y1,ans1))
function[bool]=iseven(x)
%ISEVEN Tests whether the elements of an array are even
%   _________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information 
%   (C) 2001, 2004 J.M. Lilly --- type 'help jlab_license' for details
  
bool=floor(x/2)==x/2;
function[varargout] = vsum(varargin)
%VSUM  Sum over finite elements along a specified dimension.
%
%   Y=VSUM(X,DIM) takes the sum of all finite elements of X along        
%   dimension DIM. 
%                                                                         
%   [Y,NUM]=VSUM(X,DIM) also outputs the number of good data points NUM,  
%   which has the same dimension as X.                             
%
%   [Y1,Y2,...YN]=VSUM(X1,X2,...XN,DIM) also works.
%
%   VSUM(X1,X2,...XN,DIM);  with no output arguments overwrites the 
%   original input variables.
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2001, 2004 J.M. Lilly --- type 'help jlab_license' for details  
  
if strcmp(varargin{1}, '--t')
  vsum_test,return
end

dim=varargin{end};

for i=1:length(varargin)-1
  [varargout{i},numi{i}]=vsum1(varargin{i},dim);
end

for i=length(varargin):nargout
  varargout{i}=numi{i-length(varargin)+1};
end

eval(to_overwrite(nargin-1))

function[y,num]=vsum1(x,dim)

%find sum of all good data points
bnan=~isfinite(x);
nani=find(bnan);
clear bnan
if ~isempty(nani)
    x(nani)=0;
end
y=sum(x,dim);

%find number of good data points
if ~isempty(nani)
    x(nani)=nan;
end
clear nani
x=~isfinite(x);
x=~x;
num=sum(x,dim);

index=find(num==0);
if ~isempty(index)
    y(index)=nan;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[]=vsum_test
x1=[1 2 ; nan 4];
x2=[inf 6; nan 5];
ans1=[3 4]';
ans2=[6 5]';

vsum(x1,x2,2);
reporttest('VSUM output overwrite', aresame(x1,ans1) && aresame(x2,ans2))

x1=[1 2 ; nan 4];
ans1=[3 4]';
ans2=[2 1]';

[y1,y2]=vsum(x1,2);
reporttest('VSUM sum & num', aresame(y1,ans1) && aresame(y2,ans2))


x1=[1 2 ; 0 4];
ans1=[3 4]';
ans2=[2 2]';

[y1,y2]=vsum(x1,2);
reporttest('VSUM sum & num, no NaNs', aresame(y1,ans1) && aresame(y2,ans2))








function[fz]=pdfdivide(xi,yi,fx,fy,zi)
%PDFDIVIDE  Probability distribution from dividing two random variables.
%
%   YN=PDFDIVIDE(XI,YI,FX,FY,ZI) given two probability distribution
%   functions FX and FY, defined over XI and YI, returns the pdf FZ
%   corresponding to Z=X/Y over values ZI.
%
%   For a discussion of the algorithm, see Papoulis (1991), page 138.
%  
%   Usage: yn=pdfdivide(xi,yi,fx,fy,zi);
%    
%   'pdfdivide --f' generates a sample figure
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2001, 2004 J.M. Lilly --- type 'help jlab_license' for details    
  
  
if strcmp(xi,'--f')
    pdfdivide_fig;
    return
end
  
dx=xi(2)-xi(1);

vcolon(xi,yi,fx,fy);
zi=conj(zi(:)');


ym=yi*ones(size(zi));
%fxm=fx*[1+0*zi];
fym=fy*ones(size(zi));
zm=ones(size(yi))*zi;
zym=zm.*ym;

fxmi=jinterp(xi,fx,zym);
mat=abs(ym).*fxmi.*fym;
fz=vsum(mat*dx,1)';

dz=zi(2)-zi(1);
fz=fz./vsum(fz*dz,1);
[mu,sigma]=pdfprops(zi',fz);

function[]=pdfdivide_fig
  
%test with cauchy
dx=0.1;
dy=0.05;
dz=0.025;
s1=1;
s2=2;
xi=(-10:dx:10)';
yi=(-10:dy:10)';
zi=(-10:dz:10)';


fx=simplepdf(xi,0,s1,'gaussian');
fy=simplepdf(yi,0,s2,'gaussian');

fz0=s1.*s2./pi./(s2.^2.*xi.^2+s1.^2);
fz0=fz0./vsum(fz0*dx,1);
fz=pdfdivide(xi,yi,fx,fy,zi);

figure,plot(zi,fz),hold on,plot(xi,fx)
plot(yi,fy)
linestyle default
title('RV with green pdf divided by RV with red pdf equals RV with blue pdf')
text(4,0.4,'Green and red are Gaussian')
text(4,0.35,'Blue is Cauchy')
text(4,0.30,'Dots are from a random trial')

x1=randn(100000,1)*s1;
y1=randn(100000,1)*s2;
[fz1,n]=hist(x1./y1,(-10:.1:10));
plot(n,fz1/10000,'.')





function[y]=frac(x1,x2)
%FRAC   FRAC(A,B)=A./B;
%   _________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information 
%   (C) 2004 J.M. Lilly --- type 'help jlab_license' for details  
warning('off','MATLAB:divideByZero')
y=x1./x2;
warning('on','MATLAB:divideByZero')
function[varargout]=vrep(varargin)
%VREP  Replicates an array along a specified dimension.
%
%   Y=VREP(X,N,DIM) replicates the (1-D) array by N times along 
%   dimension DIM.  For instance:   
%                                                                         
%        VREP([1:4]',3,2)=[ [1:4]' [1:4]' [1:4]' ]                            
%                                                                         
%   This is often useful in array algebra.             
%
%   [Y1,Y2,...,YP]=VREP(X1,X2,...,XP,N,DIM) also works.
%                                                                         
%   See also VINDEX, DIM.      
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2001--2008 J.M. Lilly --- type 'help jlab_license' for details    

if strcmp(varargin{1}, '--t')
  vrep_test,return
end

n=varargin{end-1};
ndim=varargin{end};

for i=1:length(varargin)-2
  varargout{i}=vrep1(varargin{i},n,ndim);
end

eval(to_overwrite(nargin-2))
 
%You would think Matlab would provide a simpler way to do this.
function[y]=vrep1(x,n,dim)
  
str='y=repmat(x,[';
ndx=ndims(x);
for i=1:max(ndx,dim)
    if i~=dim
        str=[str '1,'];
    else
	str=[str 'n,'];
    end
end
str=[str(1:end-1) ']);'];
eval(str);


function[]=vrep_test

ans1=vrep((1:4)',3,2);
ans2=[ (1:4)' (1:4)' (1:4)' ];
reporttest('VREP', aresame(ans1,ans2))

x1=(1:4)';x2=(1:4)';
vrep(x1,x2,3,2);
reporttest('VREP output redirect', aresame(x1,ans1) && aresame(x2,ans2))
function[mu,sigma,skew,kurt]=pdfprops(x,fx)
%PDFPROPS  Mean and variance associated with a probability distribution.
%
%   [MU,SIGMA]=PDFPROPS(X,FX), given a probability distribution
%   function FX over values X, returns the mean MU and the standard
%   deviation SIGMA. Each column of X must have uniform spacing.
%
%   The statistics are computed using a trapezoidal integration.
%   FX is multiplied by a constant so that it integrates to one.
%
%   [MU,SIGMA,SKEW,KURT]=PDFPROPS(X,FX) also retuns the skewness and 
%   the kurtosis, which are the third and fourth central moments, 
%   respectively normalized by the third and fourth powers of the 
%   standard deviation.  
%
%   'pdfprops --t' runs a test.
%
%   Usage:  [mu,sigma]=pdfprops(x,fx); 
%           [mu,sigma,skew,kurt]=pdfprops(x,fx);
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2001--2009 J.M. Lilly --- type 'help jlab_license' for details    
  
if strcmp(x,'--t')
    pdfprops_test;
    return
end

if jisrow(x)
  x=x(:);
end
if jisrow(fx)
  fx=fx(:);
end

if size(x,2)==1
   x=x*ones(size(fx(1,:)));
end


%dx=vrep(x(2,:)-x(1,:),size(x,1),1);
dx=x(2,:)-x(1,:);
fx=fx./vrep(trapint(fx,dx),size(x,1),1);
mu=real(trapint(fx.*x,dx));
murep=vrep(mu,size(x,1),1);
sigma=sqrt(trapint((x-murep).^2.*fx,dx));
if nargout>=3
   skew=trapint((x-murep).^3.*fx,dx);
   skew=skew./sigma.^3;
end
if nargout==4
   kurt=trapint((x-murep).^4.*fx,dx);
   kurt=kurt./sigma.^4;
end


% for i=1:size(fx,2)
%     if trapint(fx(:,i),dx(i))~=1
%         %disp('Normalizing FX to unit area.')
%         fx(:,i)=fx(:,i)./trapint(fx(:,i),dx(i));
%     end
% end
% 
% for i=1:size(fx,2)
%   mu(i,1)=real(trapint(fx(:,i).*x(:,i),dx(i)));
%   sigma(i,1)=sqrt(trapint((x(:,i)-mu(i,1)).^2.*fx(:,i),dx(i)));
% end
% 


function[y]=trapint(f,dx)
%Trapezoidal integration

fa=f;
fb=vshift(fa,1,1);
fa(1,:)=0;
fb(1,:)=0;
fa(end,:)=0;
fb(end,:)=0;
y=vsum(frac(fa+fb,2),1).*dx;
vswap(y,0,1e-10);

function[]=pdfprops_test
x=(-30:.001:30)';

mu0=2;
sigma0=5;

f=simplepdf(x,mu0,sigma0,'gaussian');  %f=simplepdf(x,mu,sig,flag)  
[mug,sigmag,skewg,kurtg]=pdfprops(x,f);

f=simplepdf(x,mu0,sigma0,'boxcar');  %f=simplepdf(x,mu,sig,flag)  
[mu,sigma]=pdfprops(x,f);
tol=1e-3;

bool(1)=aresame(mu,mu0,tol).*aresame(sigma,sigma0,tol);
bool(2)=aresame(mug,mu0,tol).*aresame(sigmag,sigma0,tol);
bool(3)=aresame(skewg,0,tol).*aresame(kurtg,3,tol);

reporttest('PDFPROPS with uniform pdf', bool(1));
reporttest('PDFPROPS with Gaussian pdf', bool(2));
reporttest('PDFPROPS Gaussian skewness=0, kurtosis=3', bool(3));

% %/********************************************************
% x=(-10:.001:10)';
% f=simplepdf(x,0,2,'gaussian');
% f(end/2:end)=2*f(end/2:end);
% f(1:end/2)=0;
% f=f./sum(f)./0.001;
% plot(x,cumsum(f*.001))
% %********************************************************

function[varargout]=vswap(varargin)
%VSWAP(X,A,B) replaces A with B in numeric array X
%
%   VSWAP(X,A,B) replaces A with B in numeric array X.  A and B may be
%   numbers, NAN, +/- INF, or NAN+SQRT(-1)*NAN.
%
%   [Y1,Y2,...YN]=VSWAP(X1,X2,...XN,A,B) also works.
%
%   VSWAP(X1,X2,...XN,A,B); with no output arguments overwrites the 
%   original input variables.    
%   __________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information
%   (C) 2001, 2004 J.M. Lilly --- type 'help jlab_license' for details  

if strcmp(varargin{1}, '--t')
  vswap_test,return
end
 
  
a=varargin{end-1};
b=varargin{end};

for i=1:length(varargin)-2
  x=varargin{i};
  varargout{i}=swapnum1(x,a,b);
end

eval(to_overwrite(nargin-2))  


function[x]=swapnum1(x,a,b)
    
    
if isfinite(a)
%   if a==0
%       if ~isreal(x)
%          a=0+sqrt(-1)*0;
%       end
%   end
  index=find(x==a);
else
  if isnan(a)
    index=find(isnan(x));
  elseif isinf(a)
    if a>0
        index=find(isinf(x)&x>0);
    else
        index=find(isinf(x)&x<0);
    end
  elseif isnan(real(a)) && isnan(imag(a))
    index=find(isnan(real(x))&isnan(imag(x)));
  end
end

if ~isempty(index)
    if allall(x==0|x==1)
       %Matlab, apparently, won't let you put NANs into a boolean array
       x=x+1;
       x(index)=b;
       x=x-1;
   else
       x(index)=b;
   end
end

function[]=vswap_test
x=(1:10);
ans1=[2 (2:10)];
reporttest('VSWAP num case', aresame(vswap(x,1,2),ans1))

x=[nan (1:10)];
ans1=(0:10);
reporttest('VSWAP nan case', aresame(vswap(x,nan,0),ans1))

x=[nan*(1+sqrt(-1)) (1:10)];
ans1=(0:10);
reporttest('VSWAP complex nan case', aresame(vswap(x,nan+sqrt(-1)*nan,0),ans1))

function[b]=jisrow(x)
%JISROW   Tests whether the argument is a row vector.
%   _________________________________________________________________
%   This is part of JLAB --- type 'help jlab' for more information 
%   (C) 2002-2011 J.M. Lilly --- type 'help jlab_license' for details
b=(ndims(x)==2) && size(x,1)==1 && size(x,2)>1;
