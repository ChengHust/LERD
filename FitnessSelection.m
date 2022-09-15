function [Next,FrontNo,CrowdDis] = FitnessSelection(Fitness,N)
% The environmental selection of NSGA-II for seleting solutions based on
% fitness values

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Fitness,N);
    Next = false(1,length(FrontNo));
    Next(FrontNo<MaxFNo) = true;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Fitness,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end