function result = perc_learn(p, x, c, lam, maxit)
    for i = 1 : maxit
        if perc_err(p, x, c) == 0
           break
        else
            p = perc_update(p, x, c, lam);
        end
    end    
    result = p;
end

function e = perc_err(p, x, c)
    e = (numel(find(perc_recall(p, x)~=c)) / length(c));
end


function U = perc_update(p, X, correct, lam)
    U = p;
    for i = 1 : size(X, 2)
        x = X(:,i);
        y = perc_recall(U, x);
        c = correct(i);
        U = U + [x', 1] * lam * (c - y);
    end
end