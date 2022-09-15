function LERDwC(Global)
% <algorithm> <A>
% LERD: Large-scale Multiobjective Optimization via Reformulated Decision 
% Variable Analysis without convergence optimization
%--------------------------------------------------------------------------
% The copyright of the PlatEMO belongs to the BIMK Group. You are free to
% use the PlatEMO for research purposes. All publications which use this
% platform or any code in the platform should acknowledge the use of
% "PlatEMO" and reference "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu
% Jin, PlatEMO: A MATLAB Platform for Evolutionary Multi-Objective
% Optimization, 2016".
%--------------------------------------------------------------------------

% Copyright (c) 2022-2023 Cheng He
	%% Initalization of population and structure
    [sN,N,gen]  = Global.ParameterSet(3,10,20);
	[W,Global.N] = UniformPoint(Global.N,Global.M); 
	Population  = Global.Initialization();
    Population  = EvolveByMOEAD(Global,Population,W,10);
    Par         = struct('N',N,'sN',sN,'Dec',rand(N,Global.D)>0.5, ...
                        'fit',zeros(N,2),'gen',gen);
    % Select the best several groups
    while Global.NotTermination(Population)
    % Convergence optimization 
        [Par,NewPop] = ReformulatedOptimization(Population,Par);
        Population   = EnvironmentSelection([Population,NewPop],Global.N);
        for g = 1 : Par.N
            PV = find(Par.Dec(g,:)==0);
            Population = SecondOptimization(Population,PV,0);
            drawnow();
        end
        deltaG     = ceil((Global.evaluated-Global.evaluated)/Global.N);
        Population = EvolveByMOEAD(Global,Population,W,deltaG);
    end
end

function [Population,fitness] = SparseFit(Archieve,Par,mask)
    Range       = [min(Archieve.objs,[],1);max(Archieve.objs,[],1)];
    fitness     = zeros(Par.N,2);
	[Dmin,Dmax] = deal(min(Archieve.decs,[],1),max(Archieve.decs,[],1));
    Upper       = repmat(Dmax,Par.sN,1);
    Lower       = repmat(Dmin,Par.sN,1);
    [~,index]   = min(calCon(Range,Archieve.objs));
	ArcDec      = repmat(Archieve(index).decs,Par.sN,1);

	Population = [];
    alpha = 0.25;
    for i = 1 : Par.N
        tempDec = ArcDec;    
        noise   = alpha.*unifrnd(0,1,Par.sN,GLOBAL.GetObj.D).*(Upper-Lower);
        tempDec(:,mask(i,:)) = noise(:,mask(i,:)) ;
        TEMP         = INDIVIDUAL(tempDec);
        fitness(i,1) = min(calCon(Range,TEMP.objs));
        fitness(i,2) = sum(mask(i,:));
        Population   = [Population,TEMP];
    end
end

function Population = SecondOptimization(Population,PV,flag)
% Convergence/Distribution optimization
    N            = length(Population);
    Range        = [min(Population.objs,[],1);max(Population.objs,[],1)];
	MatingPool   = TournamentSelection(2,N,calCon(Range,Population.objs));
    OffDec       = Population(MatingPool).decs;
    if flag == 0
        NewDec = GA(Population(randi(N,1,N+1)).decs);
        NewDec = NewDec(1:N,:);
    else
        next   = randi(N,1,2*N);
        NewDec = DE(Population.decs,Population(next(1:end/2)).decs, ...
                    Population(next(end/2+1:end)).decs, ...
                    {1,0.5,size(Range,2)/length(PV)/2,20});
    end
    OffDec(:,PV) = NewDec(:,PV);
    Offspring    = INDIVIDUAL(OffDec);
    Population   = EnvironmentSelection([Population,Offspring],N);
end

function [Par,TEMP] = ReformulatedOptimization(Population,Par)
    [TEMP,Par.fit] = SparseFit(Population,Par,Par.Dec>0);
    [~,FrontNo,CrowdDis] = FitnessSelection(Par.fit,Par.N);
    for i = 1 : Par.gen
        MatingPool  = TournamentSelection(2,2*Par.N,FrontNo,-CrowdDis);    
        OffMask     = BinaryVariation(Par.Dec(MatingPool(1:Par.N),:), ...
                                      Par.Dec(MatingPool(Par.N+1:end),:));
        [OffPop,MaskFit] = SparseFit(Population,Par,OffMask>0);         
        TEMP        = [TEMP,OffPop];
        fitness     = [Par.fit;MaskFit];
        PopDec      = [Par.Dec;OffMask];
        [Next,FrontNo,CrowdDis] = FitnessSelection(fitness,Par.N);
        Par.Dec     = PopDec(Next,:);
        Par.fit     = fitness(Next,:);
        drawnow();
    end
end

function Offspring = BinaryVariation(Parent1,Parent2)
% One point crossover and bitwise mutation

    [proC,proM] = deal(1,1);
    [N,D] = size(Parent1);
    k = repmat(1:D,N,1) > repmat(randi(D,N,1),1,D);
    k(repmat(rand(N,1)>proC,1,D)) = false;
    Offspring1    = Parent1;
    Offspring2    = Parent2;
    Offspring1(k) = Parent2(k);
    Offspring2(k) = Parent1(k);
    Offspring     = [Offspring1;Offspring2];
    Site = rand(2*N,D) < proM/D;
    Offspring(Site) = ~Offspring(Site);
end

function Con = calCon(Range,PopuObj)
% Calculate the convergence of each solution
    PopuObj = (PopuObj - Range(1,:)) ./ (Range(2,:)-Range(1,:)+1e-6);
    Con     = sum(PopuObj,2);
end