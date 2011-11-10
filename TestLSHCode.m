% Create some data.  First pick out the test dimensions, and the locations
% of some files.
D = 4;
PythonCmd = 'python2.6';
LSHCmd = 'lsh.py';
tmpFile = '/tmp/lshtest.out';
tmpFile2 = '/tmp/lshtest.out2';
subPlots = 1;

%%
% Now run the python command to create the data.
cmd = sprintf('%s %s -d %d -create', PythonCmd, LSHCmd, D);
fprintf('Running the command: %s\n', cmd);
system(cmd);
%%
% Now run the python command to measure the distance data
cmd = sprintf('%s %s -d %d -histogram', PythonCmd, LSHCmd, D);
fprintf('Running the command: %s\n', cmd);
system(cmd);
%%
% Load in the distance data and calculate the distance histograms.
testData = load(sprintf('testData%03d.distances', D));
nBins = 40;
[dnnHist, dnnBins] = hist(testData(:,1), nBins);
[danyHist, danyBins] = hist(testData(:,2), nBins);
if subPlots
    subplot(2,2,1);
end
plot(dnnBins, dnnHist, danyBins, danyHist);
legend('Nearest Neighbor', 'Any Neighbor');
xlabel('Distance')
ylabel('Frequency of Occurance');
title(sprintf('Distance Histogram for %d-D data', D));
%%
% Now calculate the optimum LSH parameters.
N=100000;
deltaTarget = 0.5;
r = 0;
uHash = 1;
uCheck = 1;
results = CalculateMPLSHParameters(N, ... 
    dnnHist, dnnBins, danyHist, danyBins, deltaTarget, r, uHash, uCheck);
%%
% Now let's run the W test.
w = results.exactW;
cmd = sprintf('%s %s -d %d -w %g -wTest > %s', PythonCmd, LSHCmd, ...
    D, w, tmpFile);
fprintf('Running the command: %s\n', cmd);
system(cmd);

%%
% And load the W-test results.
cmd = sprintf('egrep -v "[a-zA-Z]" %s | tail -20 > %s', tmpFile, tmpFile2);
[s,res] = system(cmd);
wTest = load(tmpFile2);


%%
% And now create the w-test plot.
if subPlots
    subplot(2,2,2);
end

semilogx(results.wList/results.dScale, results.binNnProb, ...
    wTest(:,1), wTest(:,2), 'bx', ...
    results.wList/results.dScale, results.binAnyProb, ...
    wTest(:,1), wTest(:,3), 'gx', ...
    [results.exactW, results.exactW], [0 1], 'r--');

xlabel('W');
ylabel('Probability of Collision');
legend('p_{nn}', 'p_{nn} experimental', ...
    'p_{any}', 'p_{any} experimental', ...
    'Optimum W', ...
    'Location', 'SouthEast');
title(sprintf('wTest for %d-D data', D));

%%
% Now let's run the K test
cmd = sprintf('%s %s -d %d -w %g -kTest > %s', PythonCmd, LSHCmd, ...
    D, w, tmpFile);
fprintf('Running the command: %s\n', cmd);
system(cmd);

%%
% And grab the K-test results.
cmd = sprintf('grep " 10 " %s > %s', tmpFile, tmpFile2);
system(cmd);
kTest = load(tmpFile2);

%% 
% Now plot the k-test results.
if subPlots
    subplot(2,2,3);
end

semilogy(kTest(:,2), kTest(:,4), 'bx', ...
    kTest(:,2), kTest(1,4).^kTest(:,2), 'g-', ...
    kTest(:,2), ...
        results.binNnProb(results.exactBin).^kTest(:,2), 'r--');
xlabel('Number of Projections (K)');
ylabel('Probability');
legend('Experimental Data', 'Extrapolated Prediction', ...
    'Theoretical Prediction', 'Location', 'SouthWest');
title(sprintf('kTest for %d-D data', D));

%%
% Now let's run the L test.
cmd = sprintf('%s %s -d %d -w %g -lTest > %s', PythonCmd, LSHCmd, ...
    D, w, tmpFile);
fprintf('Running the command: %s\n', cmd);
system(cmd);

%%
% And grab the L-test results
cmd = sprintf('grep " 10 " %s > %s', tmpFile, tmpFile2);
system(cmd);
lTest = load(tmpFile2);

%% 
% Now plot the results.
if subPlots
    subplot(2,2,4);
end

baseNN = results.binNnProb(results.exactBin).^lTest(1,2);
baseAny = results.binAnyProb(results.exactBin).^lTest(1,2);
semilogy(lTest(:,3), lTest(:,4), 'rx',  ...
    lTest(:,3), lTest(:,5), 'kx', ...
    lTest(:,3), (1-(1-baseNN).^lTest(:,3)), ...
    lTest(:,3), (1-(1-baseAny).^lTest(:,3)), ...
    lTest(:,3), (1-(1-kTest(8,4)).^lTest(:,3)), ...
    lTest(:,3), (1-(1-kTest(8,5)).^lTest(:,3)));

legend('p_{nn} Experimental', 'p_{any} Experimental', ...
    'p_{nn} Theory', 'p_{any} Theory',...
    'p_{nn} k-Prediction', 'p_{any} k-Prediction', ...
    'Location', 'SouthEast');
xlabel('Number of Tables (L)');
ylabel('Probability');
title(sprintf('kTest for %d-D data', D));

%%
save TestLSHCode wTest lTest kTest results D subplots ...
    dnnHist dnnBins danyHist danyBins

%%
% set(gcf,'Position', [100 100 900 800])
pictureFile = sprintf('TestLSHCode-%s.eps', date);
print('-depsc', pictureFile);
