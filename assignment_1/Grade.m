function G = Grade(scores, boundaries)
% Calculate Grades G from Scores using the Boudaries
% Grade G ; g_i = j iff boundaries_{j-1} > s_i >= boundaries_j

% ---- Check correct numbre of arguments ---------------------------
if nargin == 0
    error('You must provide argument Scores');
end
if nargin > 2
    error('You provided too much arguments');
end
if nargin == 1
    boundaries = [86, 71, 56];
end

% ---- Check correct shape and types of arguments ------------------
score_size = size(scores);
boundary_size = size(boundaries);
if ~ isvector(scores) || score_size(2) ~= 1
    error('scores must be a COLUMN vector!');
end
if ~ isvector(boundaries) || boundary_size(1) ~= 1
    error('boundaries must be a ROW vector!');
end

% ---- Calculation -------------------------------------------------
% add smallest value to boundaries for the last inteval
boundaries = [boundaries, -1];
G =  arrayfun(@(x) find(boundaries <= x, 1,'first'), scores); % find(boundaries <= scores, 1,'first')
end
