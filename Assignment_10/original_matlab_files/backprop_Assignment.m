

%% Train network

% Replace data, and labels with your data sets and label vector

data = [circ_data;star_data];
labels = [ones(n_imgs,1);2.*ones(n_imgs,1)];

% layer sizes

% Keep these the same
D = size(data, 2);
K = 2;

% Edit for changing size of hidden layer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize layer weights and biases
W1 = normrnd(0,1,[D,M]);
b1 = normrnd(0,1,[1,M]);
W2 = normrnd(0,1,[M,K]);
b2 = normrnd(0,1,[1,K]);

% create ground truth label probability
N = length(labels);
T = zeros([N,K]);
for i = 1:N
    T(i,labels(i)) = 1;
end

% Edit for changing learning rate and number of iterations %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
learning_rate = 5e-15;
n_iter = 1e4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% params
costs = [];
class_rate = [];

% compute randomly initialized cost
[output, hidden] = forward(data, W1, b1, W2, b2);
c = cost(T,output);
[~, P] = max(output,[],2);
r = classification_rate(labels,P);
disp(['START: | cost: ',num2str(c),' | classification rate: ',num2str(r)])
% run training loop
for epoch = 1:n_iter
    
    [output, hidden] = forward(data, W1, b1, W2, b2);
    
    if rem(epoch,500) == 0
        c = cost(T,output);
        [~, P] = max(output,[],2);
        r = classification_rate(labels,P);
        disp(['epoch: ',num2str(epoch),' | cost: ',num2str(c),' | classification rate: ',num2str(r)])
        costs = [costs, c];
        class_rate = [class_rate, r];
    end
    
    % update weights
    W2 = W2 + learning_rate * derivative_w2(hidden, T, output);
    b2 = b2 + learning_rate * derivative_b2(T, output);
	W1 = W1 + learning_rate * derivative_w1(data, hidden, T, output, W2);
	b1 = b1 + learning_rate * derivative_b1(T, output, W2, hidden);
 
end

% plot costs
figure, hold on

yyaxis left
plot(costs)
ylabel('network cost')

yyaxis right
plot(class_rate)
ylabel('classification rate')

xlabel('epoch')
title('MLP network training')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create MLP network

% define forward pass equation
function [Y, Z] = forward(data, W1, b1, W2, b2)
    Z = 1 ./ (1+exp((-data*W1) - b1));
    A = (Z*W2) + b2;
    expA = exp(A);
    Y = expA ./ sum(expA,2);
end

% define classification evaluation
function [class_rate] = classification_rate(Y, P)
    n_correct = sum(Y==P);
    n_total = length(Y);
    class_rate = n_correct / n_total;
end

function der = derivative_w2(Z, T, Y)
    der = Z'*(T-Y);
end

function der = derivative_b2(T, Y)
    der = sum(T-Y,1);
end

function der = derivative_w1(data, Z, T, Y, W2)
    dZ = ((T-Y)*W2') .* Z .* (1-Z);
    der = data'*dZ;
end

function der = derivative_b1(T, Y, W2, Z)
    der = (T-Y)*W2' .* Z .* (1-Z);
    der = sum(der,1);
end

function tot = cost(T,Y)
    tot = T .* log(Y);
    tot = sum(tot, 'all');
end
