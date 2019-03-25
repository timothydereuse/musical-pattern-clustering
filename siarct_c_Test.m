%% SET UP INPUT FILES

coreRoot = fullfile('', 'C:\Users\Tim\Documents\patterndiscovery\PattDisc-Jun2014',...
  'tunefamilytxt');
pattDiscOut = fullfile(coreRoot, 'examples', 'examplePattDiscOut');
%pieceName = 'mtcannpts_100off';
%pieceName = 'MTCtunefamily_Er_reed_er_e_ptset';

%get list of all text files in tune family database directory
allFilenames = dir(coreRoot);
allFilenames = {allFilenames(3:end).name};

%% PARAMETERS

targetPitchStd = 1.4; %for k-means clustering of mean pattern pitches

r = 1;
quick = 1;
compactThresh = 1;
cardinaThresh = 3;
regionType = 'lexicographic';
similarThresh = 0.73;
similarFunc = 'cardinality score';
similarParam = 1;
ratingField = 'cardinality';

%  r is a positive integer between 1 and n - 1, giving the number of
%   superdiagonals of the similarity array for D that will be used.
%  compactThresh is a parameter in (0, 1], giving the minimum compactness a
%   pattern occurrence must have in order to be included in the output.
%  cardinaThresh is a positive integer parameter, giving the minimum number
%   points that compactness a pattern occurrences must have in order to be
%   included in the output.
%  regionType is a string equal to 'lexicographic' or 'convex hull',
%   indicating which definition of region should be used for calculating
%   the compactness of patterns.
%  quick is an optional logical argument (set to one by default). It will
%   call a quick verison of the function in the default case, but this
%   version is sensitive to even very slight differences between decimal
%   values (look out for tuplets). The slow version is more robust to these
%   differences (down to 5 decimal places).
%  similarThresh is a value in [0, 1). If the similarity of the current
%   highest-rated pattern S_in(i) and some other pattern S_in(j) is greater
%   than this threshold, then S_in(j) will be categorised as an instance of
%   the exemplar S_in(i). Otherwise S_in(j) may become an exemplar in a
%   subsequent step.
%  similarFunc is a string indicating which function should be used for
%   calculating the symbolic music similarity, either 'cardinality score'
%   or 'normalised matching score'.
%  similarParam is an optional argument. If similarFunc = 'cardinality
%   score', then similarParam takes one of two values (one if calculation
%   of cardinality score allows for translations, and zero otherwise). If
%   similarFunc = 'normalised matching score', then similarParam takes a
%   string value ('normal', 'pitchindependent',
%   'tempoindependent', or 'tempoandpitchindependent', see fpgethistogram2
%   for details).
%  ratingField is an optional string indicating which field of each struct
%   in S should be used to order the repeated patterns.

%% RUN THE DANG THING
for fileNum = 1:3%length(allFilenames)

    pieceName = allFilenames{fileNum};
    %remove the .txt from the end
    pieceName = extractBefore(pieceName,length(pieceName)-3);

    % load piece
    piecePath = fullfile(coreRoot, [char(pieceName) '.txt']);

    % INPUT
    % D is an n x k matrix representing a k-dimensional set of n points.
    D = lispStylePointSet2Matrix(piecePath,4);
    D = D(:,[4 1 2]); %keep song ID#, onset time, and MIDI pitch number
    D = sortrows(D);

    disp(strcat('Finding Patterns in ' + pieceName + '.txt.'));

    disp('Running SIAR...');
    [SIARoutput, runtime, FRT] = SIAR(D, r, quick);

    disp('Running SIARCT...');
    [SIARCToutput, runtime, FRT] = SIARCT(D, r, compactThresh, cardinaThresh,...
    regionType,SIARoutput, runtime, FRT, quick);

    disp('Running SIARCT_C...');
    [SCout, runtime2, FRT2] = SIARCT_C(D, r,...
      compactThresh, cardinaThresh, regionType, similarThresh, similarFunc,...
      similarParam, ratingField, SIARCToutput, runtime, FRT);

    %remove identical patterns from the categorymembers field of each struct
    %i'd love to figure out why this is a thing that happens...
    disp('Removing Duplicate Pattern Occurrences...');
    SCoutSave = SCout;
    for j = 1:length(SCout)

        members = SCout(j).categoryMembers;
        %don't bother looking for duplicates if there's only one member
        if(length(members) < 2)
            continue;
            
        end

        include = zeros(1,length(members));

        for x = 1:length(members)
            includeMember = 1;
            for y = (x + 1):length(members)

                %if the patterns are different sizes, they're not equal,
                %so continue
                if any(size(members(x).pattern) ~= size(members(y).pattern))
                    continue;
                end

                %if the patterns are the same size, compare them
                %if they're equal then remove the pattern at x from
                %consideration
                temp = (members(x).pattern == members(y).pattern);
                if all(temp(:))
                    includeMember = 0;
                end
            end
            include(x) = includeMember;
        end

        SCout(j).categoryMembers = members(include > 0);
    end

    % PROBLEM: transposition is not in MTC-ANN but is considered valid pattern
    % repetition in this dataset. we need to split up these categories into
    % untransposed subcategories.

    %objective is to use kmeans clustering to break up pattern categories
    %found by SIARCT-CFP into blocks that are not transposed with respect
    %to one another. this is necessary because MTC-ANN does not use
    %transposition invariance between pattern occurrences while the SIA family
    %generally does and this is a whole lot easier than trying to tweak SIA
    sepCategories = [];

    disp('Separating Transpositions...');
    for j = 1:length(SCout)

        % ctg is current output pattern class
        ctg = SCout(j);

        %store mean pitch value for every member of this class
        pitchMeans = zeros(length(ctg.categoryMembers),1);

        for n = 1:length(ctg.categoryMembers) 
            mem = ctg.categoryMembers(n).pattern;
            pitchMeans(n) = mean(mem(:,3));
        end

        %now THIS is adaptive k-means clustering!
        %perform kmeans clustering on the pitch means until the stdv
        %between clusters is below a certain threshold
        k = 0;
        curMaxStd = 0;
        targetStd = targetPitchStd;
        maxNumClusters = 10;

        for k = 1:maxNumClusters
            meanClustered = kmeans(pitchMeans,k);      
            curMaxStd = 0;

            %find what the maximum deviation is across all clusters
            for ind = 1:max(meanClustered)
               curClusterStd = std(pitchMeans(meanClustered == ind));
               curMaxStd = max(curMaxStd,curClusterStd);
            end

            %test to see if this clustering is good. if the largest range is
            %too large then we try clustering again with a higher number
            %of clusters until the range is under control.
            if curMaxStd < targetStd
                break;
            end
        end

        %now meanClustered contains indices that separate out the motifs
        %in the current category by pitch. now, we need to insert these
        %new separated categories into a final object
        for ind = 1:max(meanClustered)
           newCat = SCout(j).categoryMembers(meanClustered == ind);   
           for ctemp = 1:length(newCat) 
                newCat(ctemp).parentCategoryIndex = j;
           end

           %reject any pattern categories of tiny size
           if(length(newCat) > 5)
                sepCategories{length(sepCategories) + 1} = newCat;
           end
        end

    end

    %sort patterns by how many times they occur
    occs = zeros(2,length(sepCategories));
    for j = 1:length(sepCategories)
        occs(1,j) = length(sepCategories{j});
        occs(2,j) = sepCategories{j}(1).cardinality;
    end

    [~,inds] = sort(occs(1,:));
    inds = fliplr(inds);
    sepSorted = sepCategories(inds);

    %now we have a bunch of patterns all cleaned up - write them into a file
    disp('Writing To File...');
    %outFilename = strcat(pieceName,'_patterns.txt');
    outFilename = [char(pieceName) '_patterns.txt'];
    fileID = fopen(outFilename,'w');
    for j = 1:length(sepSorted)
       members = sepSorted{j};
       patClassName = strcat(pieceName,'_p',string(j));
       for ind = 1:length(members)  
           curPat = members(ind).pattern;

           parentSongInd = curPat(1,1);
           parentSong = (D(D(:,1) == parentSongInd,:));

           %onset time of first note
           startOnset = curPat(1,2);
           
           
           %index of first onset of this pattern in parent song
           startOnsetInd = find(parentSong(:,2) == startOnset);
           endOnsetInd = startOnsetInd + (size(curPat,1) - 1);
           
           %subtract one from both because matlab is 1-indexed
           startOnsetInd = startOnsetInd - 1;
           endOnsetInd = endOnsetInd - 1;
           
           %for testing purposes, add the pattOccName to the struct
           patOccName = strcat(patClassName,'_o',string(ind)); 
           sepSorted{j}(ind).patternOccName = patOccName;
           
           fprintf(fileID,'%s, %s, %d, %d, %d\n',patClassName,patOccName,parentSongInd,startOnsetInd,endOnsetInd);
       end
    end
    fclose(fileID);

    %% PLOT SOME OF THE RESULTS
    % plx = 1:length(occs);
    % plot(plx,occs(1,inds),plx,occs(2,inds));
    % legend('occurrences','cardinality');
    % 
    %pattern index
%     ind = 22;
%     figure;
%     hold on;
%     for j = 1:length(sepSorted{ind})
%         pat = sepSorted{ind}(j).pattern;
%         plot(pat(:,2),pat(:,3) + (0.001 * j),'-o');
%     end

end