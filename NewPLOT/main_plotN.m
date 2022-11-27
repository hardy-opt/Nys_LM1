function [] = main_plotN()
    close all;
    % Markers = {'o','+','*','.','x','_','|','square','diamond','^','v','>','<','pentagram' }
    % To change X and Y axis properties GOTO LINE 44
    % To change dataset : Go to line 65
    %Relativer error LINE 219
    RUNS = 1;
    EPOCHS = 20;
    lambdas = [1e-3,1e-4];%1e0 1e-2 1e-4];

    etas = [ 1 0.1 0.01 0.001];%[10 1e0 1e-1 1e-2 1e-3 1e-4 1e-5 ];
    rhos = [1];%[100 1e1 1e0 1e-1 1e-2 1e-3 1e-4 1e-5]; 
    d = 4; %dataet number from followimg list

   % deltas = [100 1e1 1e0 1e-1 1e-2 1e-3 1e-4 1e-5];
   deltas=[1 0.5 0.1];
    COLS = [100];%[-1 10 50 100 500];
    BSS = [128];
    path = 'results_Sep22/';
%     path = 'result_s2qn/';
    
    datasets = {         % COL
        'REALSIM'   %1
        'CIFAR10B'  %2
        'MNISTB'    %3
        'EPSILON'   %4
        'ADULT'     %5 ---> 10
        'W8A'       %6 ---> 20
        'ALLAML'    %7
        'GISETTE'   %8 ---> 50 
        'MRI'       %9
        };

    lw = RUNS;
    ms = 8;
    params = initN(lw, ms, lambdas, etas, rhos, RUNS, EPOCHS, COLS(1), BSS(1),deltas);
%     a=params('SGD');
%     a.etas=[1e0];
%     a.marker='d';
%     params('SGD')=a;
%     b=params('SVRG');
%     b.etas=[1e-3];
%     params('SVRG')=b;
     sparams = {
      %  params('NEWTON')
%         params('Structured_QN')
%         params('Structured_QN')
%         params('OBFGS')
%         params('NSVRG')
%         params('NSGD-LM')      
          params('SVRG-LBFGS')
          params('SVRG-SQN')
           params('SQN')
          params('SGD')
%           params('SVRG')
          params('adam')
         params('NSGD')
        params('NSVRG')
       % params('NSVRG-LM')
        };
    
    yparams = {'cost', 'val_cost', 'acc_tr', 'acc_val'};
    xparams = {'time', 'epoch'};
    plot_params.sort = yparams{1};
    plot_params.y = yparams{1};
    plot_params.x = xparams{2};
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %To change the data enter the number from the data list of that dataset
    
    for dsi=d %6
            disp(datasets{dsi})
    for l = 1 :length(lambdas)
%          sparams{9}.etas=etas(l);
%          sparams{1}.etas=etas(4);
         %sparams{2}.rhos=rhos(l);
         %subplot(1,4,l);

        figure;
        plot_method_lambda(strcat(path, datasets{dsi}, '/'), sparams, lambdas(l), plot_params, 0);
%         xlim([0,200]);
%         saveas(gcf,strcat('s4qn_',lower(datasets{dsi}),'_vtc_lam',num2str(l+1),'.eps'),'epsc');
 %       legendmarkeradjust(20,20);
        fprintf('\n');
    end
    end
%    legend({'S4QN-N', 'S4QN-F', 'SVRG-LBFGS', 'OBFGS', 'SVRG-SQN', 'SQN', 'ADAM', 'NSGD-D', 'NSGD', 'NSVRG', 'NSVRG-D'})
% legend({'S4QN-N', 'S4QN-F', 'NSVRG-D', 'NSGD'});
end

function plot_method_lambda(dataset, sparams, lambda, plot_params, ref)
    hold on;
    set(gca, 'FontSize', 18);
    title(strcat('\lambda=',sprintf('10^{%0.0f}', log10(lambda))));
    switch plot_params.x
        case 'time'
            xlabel('Time (seconds)');
        case 'epoch'
            xlabel('Epochs');
    end
    switch plot_params.y
        case 'cost'
            ylabel('Opt. Error (log scale)');
        case 'val_cost'
            ylabel('Test Error (log scale)');
        case 'acc_tr'
            ylabel('Train Accuracy');
        case 'acc_val'
            ylabel('Test Accuracy');
    end
    if ref>=0
        opt_cost=inf;
        for m = 1:length(sparams)
            hold on;
            [bestmu, bestsg, besteta, bestrho, bestdelta] = find_bestN(dataset, sparams{m}.name, lambda, sparams{m}.etas, sparams{m}.rhos, sparams{m}.EPOCHS, sparams{m}.RUNS, sparams{m}.COL, sparams{m}.BS, plot_params.sort,sparams{m}.deltas);
            if bestmu.Count > 1
                y = bestmu(plot_params.y);
                if strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost')
                    opt_cost = min(opt_cost,min(y));
                end
            end
        end
    end
    maxy = -inf;
    miny = inf;
    for m = 1:length(sparams)
        hold on;
        [bestmu, bestsg, besteta, bestrho, bestdelta] = find_bestN(dataset, sparams{m}.name, lambda, sparams{m}.etas, sparams{m}.rhos, sparams{m}.EPOCHS, sparams{m}.RUNS, sparams{m}.COL, sparams{m}.BS, plot_params.sort,sparams{m}.deltas);
        if bestmu.Count > 1
            if strcmp(plot_params.y, 'val_cost') || strcmp(plot_params.y,'acc_val')
                x=bestmu(plot_params.x);
                x=x(2:end);
            else
                x=bestmu(plot_params.x);
            end
            if ref>=0 && (strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost'))
                y = (bestmu(plot_params.y)-opt_cost+ref)/(1+opt_cost);
                s = bestsg(plot_params.y)/(1+opt_cost);
            else
                y = bestmu(plot_params.y);
                s = bestsg(plot_params.y);
            end
            if bestrho == -1
                displayname=strcat(strrep(sparams{m}.name, '_', '-'), ' (\eta=', sprintf('10^{%0.0f})',log10(besteta)));
            elseif bestrho == 0
                displayname=strcat(strrep(sparams{m}.name, '_', '-'), ' (\eta=', sprintf('10^{%0.0f},',log10(besteta)),' \rho=\mid\midZ\mid\mid_{F})');
            else
                displayname=strcat(strrep(sparams{m}.name, '_', '-'), ' (\eta=', sprintf('10^{%0.0f},',log10(besteta)),' \rho=', sprintf('10^{%0.0f},', log10(bestrho)),' \delta=', sprintf('10^{%0.0f})', log10(bestdelta)));
            end
            if strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost')
                [~, idx] = min(y);
                if bestrho == -1
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  %-12s  %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, ' ', plot_params.y, y(idx), idx, round(x(idx)));
                else
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  rho: %.1e delta: %.1e %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, bestrho, bestdelta, plot_params.y, y(idx), idx, round(x(idx)));
                end
                maxy = max(maxy,y(1));
                miny = min(miny,min(y));
            else
                [~, idx] = max(y);
                if bestrho == -1
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  %-12s  %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, ' ', plot_params.y, y(idx), idx, round(x(idx)));
                else
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  rho: %.1e delta: %.1e %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, bestrho, bestdelta, plot_params.y, y(idx), idx, round(x(idx)));
                end
                maxy = max(maxy,max(y));
                miny = min(miny,min(y));
            end
            idx=length(y);
            if length(sparams)==1
                displayname = strcat(displayname, '@\lambda=', sprintf('10^{%0.0f})', log10(lambda)));
               % errorbar(x(1:idx), y(1:idx), s(1:idx), 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname);
                 plot(x(1:idx), y(1:idx), 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname);
            else
               % errorbar(x(1:idx), y(1:idx), s(1:idx), 'linestyle', sparams{m}.line, 'color', sparams{m}.linecolor, 'Marker', sparams{m}.marker, 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname);
                plot(x(1:1:idx), y(1:1:idx), 'linestyle', sparams{m}.line, 'color', sparams{m}.linecolor, 'Marker', sparams{m}.marker, 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname, 'MarkerIndices', 1:1:idx);
            end
        end
    end
    if maxy > miny 
        ylim([miny, maxy]);
    end
    %l = legend('FontSize',18,'Orientation','horizontal','NumColumns',3);
    %get(legend)
    %l.Orientation = 'horizontal';
    %legendmarkeradjust(20,20);

    if strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost')
        set(gca, 'YScale', 'log');
    end
end
