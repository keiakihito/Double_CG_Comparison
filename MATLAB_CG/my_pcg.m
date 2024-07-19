%% Define the my_pcg function
function[x_sol] = my_pcg(A, b, tol, maxitr, x_0)
    % Set up return value, x_sol
    x_sol = x_0;

    Ax = A * x_0;
    % disp(Ax);

    % r <- b - Ax
    r = b - Ax;
    % disp(r);

    % d <- r
    d = r;
    % disp(d);

    % delta_new <- r^{T} * r
    delta_new = r' * r;
    disp(delta_new);

    %Save it the calculation for relative residual
    initial_delta = delta_new;

    % Set counter
    wkr = 1;

    while wkr < maxitr && delta_new > tol * tol
         fprintf("\n\n💫💫💫= = = =  Iteration %d = = = = 💫💫💫\n", wkr);

        % q <- Ad
        q = A * d;
        fprintf("\nq = \n");
        disp(q);

        % Set dot <- (d^{T} * q)
        dot = d' * q;
        % disp(dot);

        % alpha <- delta_{new} / dot
        alpha = delta_new / dot;
        fprintf("\nalpha =  %f\n", alpha);

        % x_{i+1} <- x_{i} + alpha * d
        x_sol = x_sol + alpha * d;
        fprintf("\nx_sol = \n");
        disp(x_sol);

        if (mod(wkr, 50) == 0)
            % r <- b - Ax
            r = b - A * x_sol; % Recompute residual
        else
%            fprintf("\n\n~~Before r <- r - alpha * q ~~\n");
%            fprintf("\nr = \n");
%            disp(r);
%            fprintf("\nalpha = %f", alpha);
%            fprintf("\n\nngtAlpha = %f", -alpha);
%            fprintf("\n\nq = \n");
%            disp(q);
            % r <- r - alpha * q
            r = r - alpha * q;

%            fprintf("\n\n~~After r <- r - alpha * q ~~\n");
            fprintf("\nr = \n");
            disp(r);
%            fprintf("\nalpha = %f\n", alpha);
%            fprintf("\n\nq = \n");
%            disp(q);
        end % end of if

        % delta_{old} <- delta_{new}
        delta_old = delta_new;
        fprintf("\ndelta_old =  %f\n", delta_old);

        % delta_{new} <- r^{T} * r
        delta_new = r' * r;
        fprintf("\ndelta_new =  %f\n", delta_new);

        % Calculate relateve residual
        relative_residual = sqrt(delta_new) / sqrt(initial_delta);
        fprintf("\n\nRelative residual = %f\n", relative_residual);


        % beta <- delta_{new} / delta_{old}
        beta = delta_new / delta_old;
        fprintf("\nbeta =  %f\n", beta);

        % d_{i+1} <- r_{i+1} + beta * d_{i}
        d = r + beta * d;
        fprintf("\nd = \n");
        disp(d);



        % Increment counter
        wkr = wkr + 1;
    end % end of while

    if(wkr < maxitr)
        fprintf("\n\nmy_pcg() converged at iteration %d\n", wkr-1);
    end % end of if

    itr = wkr - 1;
    fprintf("\n\nRelative residual = %f", relative_residual);
end % end of function
