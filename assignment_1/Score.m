function  S = Score(StudentResults, MaxPoints, PointsWeight)
% Calculate Score of Student
% StudentResults -  one-column cell array containing:
%   Seminar [x_1, x_2,...] -  85 == 35%
%   Tests [a,b] \in [0,10] -  20 == 20%
%   Exam                   -  45 == 45%
% MaxPoints=85      ... Seminar max points
% PointsWeight=35   ... Seminar percentage 

% ---- Check correct numbre of arguments ---------------------------
if nargin == 0
    error('You must provide argument StudentResults');
end
if nargin > 3
    error('You provided too much arguments');
end

if nargin == 1
    MaxPoints = 85;
    PointsWeight = 35;
elseif nargin == 2
    PointsWeight = 35;
end

% ---- Check correct shape and types of arguments ------------------
if ~ isscalar(MaxPoints)
    error('MaxPoints must be a scalar value');
end
if ~ isscalar(PointsWeight)
    error('PointsWeight must be a scalar value');
end
if ~ iscell(StudentResults)
    error('StudentResults must be a cell array');
end
StudentResults_size = size(StudentResults);
if StudentResults_size(2) ~= 1
    error('StudentResults must be a 1 COLUMN  cell array!');
end
% TODO check that each cell is cell array with 3 cells 
% and first two are row vectors

% ---- Calculation -------------------------------------------------
%   seminar_percentage = (min(sum(seminars), MaxPoints) * PointWeight) / MaxPoints 
%   score = seminar_percentage + sum(tests) + exam
S = arrayfun(@(result_arr)  min(sum(result_arr{1}{1}),MaxPoints)*PointsWeight/MaxPoints + sum(result_arr{1}{2}) + result_arr{1}{3} , StudentResults);
end


