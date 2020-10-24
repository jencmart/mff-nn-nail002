function P = HowManyPercent(TargetGrade, StudentResults, Boundaries, MaxPoints, PointsWeight, MaxExamScore)
% Calculate min. score which must student get form Exam to obtain TARGET_GRADE 
% we have 
%  points from  seminar 
%  percent scores from the tests

% HowManyPercent(TargetGrade,StudentResults), 
% Boundaries=[86,71,56], MaxPoints=85, PointsWeight=35 and MaxExamScore=45

% While MaxPoints, PointsWeight and MaxExamScore ... scalars
% TargetGrade and StudentResults ... same number of rows

% ---- Check correct numbre of arguments ---------------------------
if nargin < 2
    error('You must provide argument TargetGrade and  StudentResults');
end
if nargin > 6
    error('You provided too much arguments');
end
if nargin == 2
    Boundaries=[86,71,56];
    MaxPoints=85;
    PointsWeight=35;
    MaxExamScore=45;
elseif nargin == 3
    MaxPoints=85;
    PointsWeight=35;
    MaxExamScore=45;
elseif nargin == 4
    PointsWeight=35;
    MaxExamScore=45;
elseif nargin == 5
    MaxExamScore=45;
end
% ---- Check correct shape and types of arguments ------------------
targetGrade_size = size(StudentResults);
if ~ isvector(TargetGrade) || targetGrade_size(2) ~= 1
    error('TargetGrade must be a 1 COLUMN vector');
end
if ~ iscell(StudentResults)
    error('StudentResults must be a cell array');
end
StudentResults_size = size(StudentResults);
if StudentResults_size(2) ~= 1
    error('StudentResults be a 1 COLUMN  cell array!');
end
% student results and target grade must have same number of rows
if targetGrade_size(1) ~= StudentResults_size(1)
   error('TargetGrade and StudentResults must have same number of rows');
end
boundary_size = size(Boundaries);
if ~ isvector(Boundaries) || boundary_size(1) ~= 1
    error('boundaries must be a ROW vector!');
end
if ~ isscalar(MaxPoints)
    error('MaxPoints must be a scalar value');
end
if ~ isscalar(PointsWeight)
    error('PointsWeight must be a scalar value');
end
if ~ isscalar(MaxExamScore)
    error('MaxExamScore must be a scalar value');
end

% ---- Calculation -------------------------------------------------

% First calculate current score ...
curr_score = arrayfun(@(result_arr)  min(sum(result_arr{1}{1}),MaxPoints)*PointsWeight/MaxPoints + sum(result_arr{1}{2}), StudentResults);
% Target percentage is given by Boundaries(TargetGrade(student_i))
target_score = arrayfun(@(grade) Boundaries(TargetGrade(grade)), TargetGrade);
% Required points: Boundaries(TargetGrade(student_i)) - curr_score(student_i)
required_score = target_score - curr_score;
% if required score > 45  , NaN , else OK
required_score(required_score > 45) = NaN;
P = required_score;
end


