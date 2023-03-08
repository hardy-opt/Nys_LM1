function  nys_curve_expLM(darg,col,e) %4.8446772001339822e-01
     clc;
%     clear;
    close all;
    addpath('/home/hardik/data/')
    addpath(genpath(pwd));
   % addpath('/data/Datasets/'); % Dataset repository
    NUM_RUN = 3;
    NUM_EPOCH = e;
    P = 10;  %Partition for DP sampling
    K = 0;  % No. of clusters for DP sampling
    dat = strcat('Nystrom_Result23/',darg);  % result path
    method = {'NSGD', 'NGD','NGD1','NGD2', 'Nystrom_GD', 'Nystrom_GDLM','Nystrom_GD1', 'Nystrom_GDLM1','Nystrom_GD2', 'Nystrom_GDLM2' };
    omethod = {'SVRG-LBFGS', 'SVRG-SQN', 'adam', 'SQN', 'OBFGS', 'SVRG', 'SGD', 'LBFGS','GD','NG','RNGS','NEWTON'};
    %omethod = {'adam','SGD','NEWTON'};
    BATCHES = 128;% [64 128];
    COLS = col;%100];% 50];% 100]; %columns-a8a = [10,100,800], Epsilon = [100,800, 3200];
    %rho = 1;
    COLS;
    for s=1:NUM_RUN
        for reg= [1e-5 ]
            for step = [1 0.1 0.01 0.001]
                data = loaddata(s, reg, step, dat);
                for rho = [ 1]
                    for m= [1 2,3,4]%[5 6 7 8 9 10]
                        for COL =  COLS 
                            if COL > size(data.x_train,1)
                                break;
                            end
                            for del= [1 0.1 0.01 0.001]%BATCH_SIZE = BATCHES
%                                 if BATCH_SIZE > size(data.x_train,2)
%                                     break;
%                                 end
                                %rng(s); % do not remove 
                                %If you want to give more batches then use
                                %BATCHES(1) or ADD FOR LOOP
                                BATCH_SIZE = BATCHES;
                                %disp([m, method{m}])
                                fprintf('K%d - B%d - %s - Reg:%f - Step:%f - Rho:%f - Delta:%f - Run:%d\n', COL, BATCH_SIZE, method{m}, reg, step, rho, del, s);
                                options.max_epoch=NUM_EPOCH;    
                                %problem = linear_svm(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                                problem = logistic_regressionLM(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                                options.w_init = data.w_init;   
                                options.step_alg = 'fix'; 
                                options.step_init = step; 
                                options.verbose = 2;
                                options.batch_size = BATCH_SIZE;
                                options.column = COL;
                                options.partitions = P;
                                options.clusters = K;

                                Name = sprintf('%s/K%d_B%d_%s_%.1e_R_%.1e_rho_%.1e_delta_%.1e_run_%d.mat',dat, COL, BATCH_SIZE, method{m},options.step_init,reg,rho, del, s);
                                
                                if m==1
                                   % if del == 1
                                    
                                   [w_s1, info_s1] = Nystrom_sgd(problem, options,reg,del);  % NSVRG-
                                   save(Name,'info_s1');
                                   % Nystrom_svrg(problem, in_options,reg,dp)
                                   % If dp = 1 then NSVRG-DP
                                   % else NSVRG
                                   
                                   % end
                                elseif m==2
                                    
                                    %options.step_alg = 'decay-2'; %decay
                                    [w_s1, info_s1] = Nystrom_gd(problem, options,reg,del); % NSGD-
                                    save(Name,'info_s1');
                                    
                                elseif m==3
                                   [w_s1, info_s1] = Nystrom_gd1(problem, options,reg,del);  % NSVRG
                                   save(Name,'info_s1');
                                   
                                elseif m==4
                                    %options.step_alg = 'decay-2'; %decay
                                    [w_s1, info_s1] = Nystrom_gd2(problem, options,reg,del); % NSGD
                                    save(Name,'info_s1');
                                    
                                elseif m==5
                                    if del==1
                                    %options.step_alg = 'decay-2';
                                    [w_s1, info_s1] = Nystrom_gd(problem, options,reg,rho);  %Nystrom
                                    save(Name,'info_s1');
                                    end
                                elseif m==6
                                
                                    %options.step_alg = 'decay-2';
                                    [w_s1, info_s1] = Nystrom_gd(problem, options,reg,0,rho,del); %Fisher
                                    save(Name,'info_s1');
                                  elseif m==7
                                    if del==1
                                    %options.step_alg = 'decay-2';
                                    [w_s1, info_s1] = Nystrom_gd1(problem, options,reg,rho);  %Nystrom
                                    save(Name,'info_s1');
                                    end
                                elseif m==8
                                
                                    %options.step_alg = 'decay-2';
                                    [w_s1, info_s1] = Nystrom_gdLM1(problem, options,reg,0,rho,del); %Fisher
                                    save(Name,'info_s1');
                                  elseif m==9
                                    if del==1
                                    %options.step_alg = 'decay-2';
                                    [w_s1, info_s1] = Nystrom_gd2(problem, options,reg,rho);  %Nystrom
                                    save(Name,'info_s1');
                                    end
                                elseif m==10
                                
                                    %options.step_alg = 'decay-2';
                                    [w_s1, info_s1] = Nystrom_gdLM2(problem, options,reg,0,rho,del); %Fisher
                                    save(Name,'info_s1');
                                end 
                            end
                            
                        end
                    end
                end
                 for BATCH_SIZE = BATCHES
                    if BATCH_SIZE > size(data.x_train,2)
                        break;
                    end
                    for m= 7 %[8 9 12]%[1 2 4 8 12] %[1 2 4 ]%[1,2,3,4,6,7]
                        
                        fprintf('%s - Reg:%f - Step:%f  - Run:%d\n', omethod{m}, reg, step, s);
                        options.max_epoch=NUM_EPOCH;    
                        problem = logistic_regressionLM(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                        %problem = linear_svm(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                        options.w_init = data.w_init;   
                        options.step_alg = 'fix';
                        options.step_init = step; 
                        options.verbose = 2;
                        options.batch_size = BATCH_SIZE;

                        Name = sprintf('%s/B%d_%s_%.1e_R_%.1e_run_%d.mat',dat,BATCH_SIZE,omethod{m},options.step_init,reg,s);

                        if m==1    
                            options.sub_mode='SVRG-LBFGS';
                            %options.sub_mode= 'Lim-mem';
                            [w_s1, info_s1] = slbfgs(problem, options);
                        elseif m==2
                            options.sub_mode='SVRG-SQN';
                            %options.sub_mode= 'Lim-mem';
                            [w_s1, info_s1] = slbfgs(problem, options);
                        elseif m==3
                            options.step_alg = 'decay-2'; 
                            options.sub_mode='Adam';
                            [w_s1, info_s1] = adam(problem, options);
                        elseif m==4
                            options.store_grad = 0;
                            options.sub_mode = 'SQN';
                            options.step_alg = 'decay-2';
                            [w_s1, info_s1] = slbfgs(problem, options);
                        elseif m==5
                            options.sub_mode = 'Lim-mem';
                            [w_s1, info_s1] = obfgs(problem, options);
                        elseif m==6
                             %options.step_alg = 'decay-2'; 
                             [w_s1, info_s1] = svrg(problem, options);
                        elseif m==7
                            options.step_alg = 'decay-2'; 
                            [w_s1, info_s1] = sgd(problem, options);
                       
                        elseif m==8
                            
                            options.sub_mode = 'STANDARD';
                            %options.regularized = true;
                            options.step_alg = 'backtracking';
                            %options.max_epoch=5;
                            [w_s1, info_s1] = lbfgs(problem, options);
                            
                         elseif m==9 %Gradient Descent
                            
                             options.sub_mode = 'STANDARD';
                            %options.regularized = true;
                            options.step_alg = 'backtracking';
                            %options.max_epoch=5;
                            [w_s1, info_s1] = grd(problem, options);
                         
                            
                            
                          elseif m==10
                            options.sub_mode = 'STANDARD';
                            %options.regularized = true;
                            options.step_alg = 'backtracking';
                            %options.max_epoch=5;
                            [w_s1, info_s1] = ng(problem, options);
                         elseif m==11
                            options.sub_mode = 'STANDARD';
                            %options.regularized = true;
                            options.step_alg = 'backtracking';
                            %options.max_epoch=5;
                            [w_s1, info_s1] = rngd(problem, options,col);
                        
                         elseif m==12
                            %options.max_epoch=15;    
                            options.sub_mode = 'STANDARD';
                            %options.sub_mode = 'CHOLESKY';
                            %options.regularized = true;
                            options.step_alg = 'backtracking';
                            %options.max_epoch=5;
                            [w_s1, info_s1] = newton(problem, options);
                            
                        end                    
                        save(Name,'info_s1');
                    end
                end
            end
        end
    end
end

function [data]=loaddata(s,reg,step,dat)
    strs = strsplit(dat,'/');
    if strcmp(strs{end}, 'REALSIM')
        data = REALSIM(s,reg,step);
    elseif strcmp(strs{end}, 'CIFAR10B')
        data = CIFAR10B(s,reg,step);
    elseif strcmp(strs{end}, 'MNISTB')
        data = MNISTB(s,reg,step);
    elseif strcmp(strs{end}, 'EPSILON')
        data = EPSILON(s,reg,step);
    elseif strcmp(strs{end}, 'ADULT')
        data = ADULT(s,reg,step);
    elseif strcmp(strs{end}, 'W8A')
        data = W8A(s,reg,step);
    elseif strcmp(strs{end}, 'ALLAML')
        data = ALLAML(s,reg,step);
    elseif strcmp(strs{end}, 'SMK_CAN')
        data = SMK_CAN(s,reg,step);
    elseif strcmp(strs{end}, 'GISETTE')
        data = GISETTE(s,reg,step);
    elseif strcmp(strs{end}, 'A8AN')
        data = A8A(s,reg,step);
    elseif strcmp(strs{end}, 'MRI')
    data = MRI(s);
    elseif strcmp(strs{end}, 'IJCNN')
        data = IJCNN(s,reg,step);
    else
        disp('Dataset tho de');
    end
end
