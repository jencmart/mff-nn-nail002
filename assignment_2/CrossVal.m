function [delta,s] = CrossVal(Train,Predict,Params,Train2,Predict2,Params2,X,y,k,NoShuffle)

    % Basic argumet checks ...
    if nargin < 9
        error('You must provide Train,Predict,Params,Train2,Predict2,Params2,X,y,k');
    elseif nargin > 10
        error('You must not provide anything but Train,Predict,Params,Train2,Predict2,Params2,X,y,k, {NoShuffle}');
    end
    
    % if NoShuffle is not present, do the random permutation...
    if nargin == 9
        permutation = randperm(length(y));
        X=X(:,permutation);
        y=y(permutation);
    end
    
    % Perform the k-fold cross validation...
    errors1 = [];
    errors2 = [];
    for test_idx = 1 : k
        % Split to train and test sets 
        [train_X, train_y, test_X, test_y] = train_test_split(test_idx, X, y, k);
        % Error of algorithm 1
        E1 = Err(Train, Predict, Params, train_X, train_y, test_X, test_y);
        % Error of algorithm 2
        E2 = Err(Train2, Predict2, Params2, train_X, train_y, test_X, test_y);

        % Append the errors...
        errors1 = [errors1 E1];
        errors2 = [errors2 E2];
    end
    
    % calculate mean and standard deviation
    deltas = errors1 - errors2;
    delta = mean(deltas);
    s = std(deltas);
end

function [train_X, train_y, test_X, test_y] = train_test_split(test_idx, X, y, k)
    % calculate size of the bin
    % note that last bin may be a bit smaller if len % k != 0
    % i.e.  last_bin_size = len - bin_size*(k-1);
    len = length(y);
    bin_size = ceil(len/k);

    % idx for better calculation of the positons
    idx = test_idx - 1;

    % Create Testing set
    test_X = X(:, idx*bin_size + 1 :  min( idx*bin_size + bin_size, len));
    test_y = y(idx*bin_size + 1 :  min( idx*bin_size + bin_size, len));

    % Create Training set
    if test_idx == 1
        train_X = X(:, min( idx*bin_size + bin_size +1, len) : len);
        train_y = y(min( idx*bin_size + bin_size +1, len) : len);
    elseif test_idx == k
        train_X = X(:, 1 : idx*bin_size);
        train_y = y(1 : idx*bin_size);
    else
        before_X = X(:, 1: idx*bin_size);
        before_y = y(1 : idx*bin_size);
        after_X = X(:, min( idx*bin_size + bin_size +1, len) : len);
        after_y = y(min( idx*bin_size + bin_size +1, len) : len);
        train_X = [before_X after_X];
        train_y = [before_y after_y];
    end
    
end