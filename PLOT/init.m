function [methods] = init(lw, ms, lambdas, etas, rhos, RUNS, EPOCHS, COL, BS)
    methods=containers.Map;

    params.name = 'SGD';
    params.line = ':';
    params.linewidth = lw;
    params.facecolor = 'none'; %[1 0 0];
    params.linecolor = [0 0.5 1];
    params.marker = '*';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    params.name = 'SVRG';
    params.line = ':';
    params.linewidth = lw;
    params.facecolor = 'none'; %[1 0 0];
    params.linecolor = [0 0 1];
    params.marker = 'diamond';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    params.name = 'adam';
    params.line = ':';
    params.linewidth = lw;
    params.facecolor = 'none'; %[1 0 1];
    params.linecolor = [0.5 0 1];
    params.marker = 'p';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    params.name = 'SVRG-LBFGS';
    params.line = ':';
    params.linewidth = lw;
    params.facecolor = 'none'; %[0 0 1];
    params.linecolor = [0 0 1];
    params.marker = '^';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    params.name = 'OBFGS';
    params.line = ':';
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [0 .8 1];
    params.marker = 'p';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    params.name = 'SVRG-SQN';
    params.line = ':';
    params.linewidth = lw;
    params.facecolor = 'none'; %[0 0 0];
    params.linecolor = [0 0.5 0];
    params.marker = '>';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    params.name = 'SQN';
    params.line = ':'; %c
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [0 0.8 0.35];
    params.marker = 'p';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    %Proposed
    params.name = 'NSVRG2';
    params.line = '-';
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [0 0.1 0.6];
    params.marker = 'o';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    params.name = 'NSVRG';
    params.line = '--';
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [0.75 0 0];
    params.marker = 's';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    params.name = 'NSGD-LM';
    params.line = '-';
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [.3 0.5 0.8];
    params.marker = 'o';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    params.name = 'NSGD';
    params.line = '--';
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [1 0.5 0.5];
    params.marker = 's';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    params.name = 'Structured_QN';
    params.line = ':';
    params.linewidth = lw;
    params.linecolor = [0 0 1];
    params.marker = '*';
    params.markersize = ms;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    params.name = 'Structured_QF';
    params.line = ':';
    params.linewidth = lw;
    params.linecolor = [0 1 1];
    params.marker = 's';
    params.markersize = ms;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;

    params.name = 'NEWTON';
    params.line = '-.';
    params.linewidth = lw;
    params.linecolor = [0.25 0.25 0.75];
    params.marker = 'p';
    params.markersize = ms;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    params.COL = COL;
    params.BS = BS;
    methods(params.name)=params;
end
