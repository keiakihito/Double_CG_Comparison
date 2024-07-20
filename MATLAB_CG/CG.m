%Display up to 1e-8 point
format long g

% Main script starts here
% Set up 3 x 3 SPD matrix hardcoded
fprintf('~~5 x 5  SPD matrix~~ \n');
A = [
    1.5004, 1.3293, 0.8439; 1.3293, 1.2436, 0.6936; 0.8439, 0.6936, 1.2935];


A =[10.840188, 0.394383, 0.000000, 0.000000, 0.000000;
        0.394383, 10.783099, 0.798440, 0.000000, 0.000000;
        0.000000, 0.798440, 10.911648, 0.197551, 0.000000;
        0.000000, 0.000000, 0.197551, 10.335223, 0.768230;
        0.000000, 0.000000, 0.000000, 0.768230, 10.277775];
% disp(A);

% Set given vector b = [1, 1, 1]
bT =[-0.957936, 0.099025, -0.312390, -0.141889, 0.429427];
b = bT';
%disp(b);

% Set up initial guess x_0 = [0, 0, 0]
x_0 = b;
%disp(x_0);

% Set epsilon
eps = 1e-6;

% Max number of iterations
maxItr = 5;

% Solve Ax = b with manual CG implementation
fprintf('Solve Ax = b with my_pcg()\n');
x_myPcg = my_pcg(A, b, eps, maxItr, x_0);
fprintf("\n\nx_sol = \n");
disp(x_myPcg);


% Answer key
% Solve Ax = b with pcg
fprintf('\n~~Answer Key~~\n');
fprintf('Solve Ax = b with pcg()\n');
x_ans = pcg(A, b, eps, maxItr);
disp(x_ans);

% Compare answer and my solution
validateSol(x_ans, x_myPcg);
















