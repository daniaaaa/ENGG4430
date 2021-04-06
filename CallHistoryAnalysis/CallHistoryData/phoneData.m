% TO DO:
% visualization of sus vs clear numbers (noe for every phone number, with connections given by rows of callData)
% writing data to output files
% playing with the tuning parameters?
%file size = 1.29994*exp(noNums/1000) - 1.59405

clear;
noNums = 10; %number of phone numbers to generate, defines dataset size
noSusNums = round(0.3*noNums); %proportion of sus numbers a fraction of the total noNums
numData = ones(noNums, 1)*NaN; %stores phone numbers
callData = ones(noNums, noNums)*NaN; %stores number of calls made
AreaCodes =  [226,249,289,343,365,416,437,519,548,613,647,705,807,905,888,800,1800]*1e+7; 

%for non-suspicious ("Clear") numbers:
maxClearCalls = 10; maxClearCallProb = zeros(1,maxClearCalls); maxClearCallProb(1,maxClearCalls) = 1; %defines overall call frequency of non sus numbers (each number only has a small [1/maxCalls] chance of making a call to another
%larger number is less frequent calling, smaller # is more frequent calling

%for suspicious ("Sus") numbers:
maxSusCalls = 2; maxSusCallProb = zeros(1,maxSusCalls); maxSusCallProb(1,maxSusCalls) = 1; %defines overall call frequency of non sus numbers (each number only has a small [1/maxCalls] chance of making a call to another
%larger number is less frequent calling, smaller # is more frequent calling, for example =2 implies that scam number called half the numbers in the
%list


%generate suspicious numbers first
for q = 1:1:noSusNums %each row of column 1 = phone number
    numData(q,1) = AreaCodes(1,randi([1,length(AreaCodes)])) + randi([1e+6, 9999999],1);
    for t = 1:1: noNums %column index, ignores first (phone number) index
        callData(q,t) =  maxSusCallProb(1,(randi([1,length(maxSusCallProb)],1)));
    end
end

%generating non suspicious numbers
for q = noSusNums+1:1:noNums %row index
    numData(q,1) = AreaCodes(1,randi([1,length(AreaCodes)])) + randi([1e+6, 9999999],1);
    for t = 1:1: noNums %column index, ignores first (phone number) inde
       callData(q,t) =  maxClearCallProb(1,(randi([1,length(maxClearCallProb)],1)))*randi([1,5],1); 
    end
end

% num2str(numData) 
% callData
% csvwrite('trainData.csv', callData);
% outputInfo = dir('trainData.csv'); 
% fileSize = outputInfo.bytes/(1e+6);
% fprintf('approximately %4.2f MB wirtten\n', fileSize);
outputData = zeros(noNums,5);
outputData(:,1) = numData;
for q = 1:1:noNums
    for t = 1:1:4
        outputData(q,2) = sum(callData(q,:),'omitnan'); %total calls made
        outputData(q,3) = noNums - count(num2str(callData(q,:)),"0"); %number of unique numbers called
        outputData(q,4) = outputData(q,2)/outputData(q,3);
    end
end

outputData([1:noSusNums],5)=1;
outputData([noSusNums+1:noNums],5)=0;
format longg
% csvwrite('outputData.csv', outputData);
% outputInfo = dir('outputData.csv'); 
% fileSize = outputInfo.bytes/(1e+6);
% fprintf('approximately %4.2f MB written\n', fileSize);

outputData