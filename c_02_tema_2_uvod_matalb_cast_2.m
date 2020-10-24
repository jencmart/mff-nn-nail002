

%% A Basic Introduction into MATLAB - part 2
% This file is formatted in such a way that it can serve for generating HTML-file
% with code blocks while showing also the output of the included commands. 
% Some symbols (like stars, underline symbols, etc.)
% within the comments in the source code (prefixed by the percent sign)
% serve for formatting the PDF- or HTML-output of the script. 
%
% Moreover, the file is formatted in such a way that it is possible to 
% execute parts of the file called cells, delimited by lines which start
% with double percent sign %%.
%%
% First, let us clear the workspace - delete all defined variables. 
clear
%% Sparse matrices
% For matrices with only a few non-zero elements, it is advantageous to store only 
% the non-zero elements. Such representation is called a sparse matrix. 
% Sparse matrices represent in memory only the non-zero elements, hence they 
% can be memory efficient. In the following example |A| is an ordinary 
% matrix which we will convert into its sparse variant |B|:
A=[ 0 2 0; 1 0 0; 0 0 5]
%%
B=sparse(A)
%%
whos
%%
% Any sparse matrix can be used in any expression where ordinary matrix can be 
% used. MATLAB can compute with sparse matrices in the same way as with ordinary 
% matrices. Both representations of matrices can be arbitrarily mixed in any 
% expression:
C=3*B
%%
D=A+B
%%
% We can convert a sparse matrix into an ordinary matrix called "full" matrix:
E=full(B)
%%
% In MATLAB, there are several functions which generate sparse matrices directly.
% (e.g. |speye(n)|, see |help|) but we will not need them
%% Structures
% A structure corresponds to |struct| from the C-language but without
% declaring its
% fields. Fields of a structure are accessed using the dot-notation.
a.x=8
%%
a.y=9
%%
sqrt(a.x^2+a.y^2)
%%
whos
%%
fieldnames(a)
%% Cell arrays
% An ordinary matrix (an $n$-dimensional array) in MATLAB is homogenous, i.e. all its 
% components are of the same type. This does not allow to store entries of 
% different types in a single array. However, exactly this is possible with cell arrays.
% In contrast to ordinary arrays, cell arrays are written surrounded by 
% parentheses '{' and '}' and also their indexes are written within the same 
% type of parentheses:
A = {1 [3;2] [1 2; 3 4]}
%%
A{2}
%%
whos 
%% Tables
% In contrast to cell arrays, tables are arrays where date in one column
% are of the same type while data in different columns can be of different
% types. 
% Often they are handy when we import CSV-files where some columns are
% numerical and other are, e.g., textual or categorical. More on tables can
% be found in help.
%% Classes
% MATLAB and also Octave enable to define classes and work with them in an 
% object oriented way (see |help| for |class|). Nevertheless we will not use them.

%% The Programming Language of MATLAB
% MATLAB is an expression oriented language. Entered expressions are immediately 
% evaluated by an interpreter. Most commands are of the form 
%
% _variable_ = _expression_
%
% or directly
% 
% _expression_

%%
% Each line can contain more than one command separated by commas or semicolons.
% An expression followed by a semicolon is evaluated, but its result is not 
% displayed. This is used usually in scripts and functions in order not to display 
% all intermediate results.
b=eye(3); c = 2*b; d = c^4+b
%%
% Structured commands like cycles can be used also in the interactive mode, but
% usually they are used in scripts (programs and functions). Let us look at 
% cycles first. |for|-cycles are very common:
for I = 1:10
    fprintf('%d^2 = %d\n',I,I*I)
end
%%
% Also |while|-cycles are possible:
I=1;
while I<=10
    x=[I,I*I]
    I=I+1;
end
%%
% |if| statement is rather general
%
% |if| _condition_
%
%     commands
%
% |elseif| _condition_
%
%     Commands
%
% ...
%
% |else|
%
%    commands
%
% |end|
I = 6;
if I<2
    disp('Not enough')
elseif I<10
    disp('It''s O.K.')
else
    disp('This is too much')
end
%%
% In MATLAB, relational operators are
%
% * <     less
% * >     greater
% * <=    less or equal
% * >=    greater or equal
% * ==    equal
% * ~=    not equal
%
% Conditions can be joined using logical operators
%
% * &     and
% * |     or
% * ~     not (negation)
x = randn(1)
%%
if -1<x & x<1
    disp('Almost zero')
end
%%
% *Be careful when comparing matrices!* If we want to execute a command
% if two matrices are equal:
A = rand(2);
B = A;
if A==B
    disp('They are the same')
end
%%
% But even small difference changes the outcome.
A(1,1)=2
B(1,1)=sqrt(2)^2;
if A==B
    disp('They are the same')
else
    disp('They are different')
end
%%
% A better comparison of real-valued matrices:
tol=0.00001
if abs(A-B)<abs(A)*tol
    disp('They are the same')
else
    disp('They are different')
end
%%
% Be careful when testing for inequality of matrices
if A~=B
    disp('They are different')
else
    disp('They are the same')
end
%%
% The test |A~=B| returns a matrix of zeros and ones. 
T = A~=B
if T
    disp('They are different')
else
    disp('They are the same')
end
%%
% The condition |if T| tests whether all entries of |T| are non-zero! 
% To do it correctly we can use the function |any(x)| that for a vector |x|
% returns 1 if at least one entry of the vector |x| is non-zero and 
% for a matrix |x| it returns a row vector of results of |any| on each 
% of the columns of |x|.
if any(any(A~=B))
    disp('They are different')
end
%%
% Or simpler
if A==B else
    disp('They are different')
end
%%
% Arrays can be reduced to vectors or scalars also by applying the function 
% |all| - see help.

%% Programming in MATLAB
% Programs in MATLAB are stored in m-files (they have the extension |.m|) and come
% in two forms 
% 
% # scripts or
% # functions.
% 
%% Scripts 
% are simply sequences of commands which should be executed sequentially.
% If a script is named |XX.m|, then is can be run by the command |XX|. 
% All variables occurring in a script are global. Hence scripts can be used 
% for initializing variables or performing some computations.
%% Functional m-files
% A functional m-file contains a definition of a single function. The name of the defined
% function must be the same as the name of the m-file with the definition. 
% the first line of the file must contain a header of the function in the
% following form 
%
% |function result = XXX(par1,par2,...,parN)|
%
% |result| is a variable which will be set to the result of the function, 
% |XXX|    is the name of the function, the function must be stored in the
%          file |XXX.m|, and
% |par1, ..., parN| are parameters of the function. The type of the 
% parameters is not specified.
% 
% A sample function with further explanations can be found in the file |GenMatAB.m|
