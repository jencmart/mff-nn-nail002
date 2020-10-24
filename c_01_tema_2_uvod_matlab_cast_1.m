%% A Basic Introduction into MATLAB
% This file is formatted in such a way that it can serve for generating
% HTML-file or PDF-file (using pdfLaTeX) with code blocks while showing
% also the output of the included commands. Some symbols within the
% comments (like stars, underline symbols, etc.) in the source code
% (prefixed by the percent sign) serve for formatting the HTML-output of
% the script.
% 
% The name MATLAB is an abbreviation of MATrix LABoratory. It is a program
% designed for matrix computations more than 40 years ago.
%% 
% The basic usage of MATLAB:
% 
% * solving systems of linear equations,
% * computing eigenvalues and eigenvectors,
% * factorizations of matrices,
% * rich graphical capabilities,
% * it has a proprietary programming language,
% * designed to solve problems numerically - not for symbolic computations
%   (but also this is now possible using special packages - toolboxes).
%
% Nowadays it is de-facto standard for technical computing:
%
% * simulations,
% * control,
% * computer vision, and
% * machine learning.
%
%% 
% The majority of functions from MATLAB is implemented also in |GNU
% Octave|. |Octave| is a free multi-platform clone of MATLAB. It is
% available both for Windows and Linux. Only recently it has got an IDE,
% formerly its graphical output was possible only through |GNU Plot|.
% 
% There are more MATLAB clones. Such clone is |freemat| also for both
% Windows and Linux. |Octave| comes also with an IDE.
%% Simple Arithmetic
% How to enter numbers, vectors and matrices, and basic arithmetic.
% 
% The basic data type in MATLAB is an n-dimensional (primarily
% 2-dimensional) array of double-precision numbers. They are written as
% sequences of numbers surrounded by brackets. Elements of a matrix can be
% separated by spaces, tabs, and commas in rows. Semicolons and end of lines
% separate lines of matrices. First, let us enter two vectors - a row
% vector |x| and a column vector |y|. By multiplying |x| and |y| we obtain
% a matrix.
x=[2,3 4]
%%
y=[1;2]
%%
z=y*x

%% 
% Entering a matrix
% 
% * on a single line, or
% * on multiple lines
A=[2.3  4.8 9.2; 21.34 1.3e12 9] 
%%
% When MATLAB displays the matrix |A|, it prints a common coefficient
% |1.0e+012 *| in the front of the matrix. All entries in the following
% matrix must be multiplied by this coefficient.
% 
B = [ 1 9 23
        21 2 1
        9 11 99]
%%
% Scalars are matrices of size |1x1|
a = 3.1
%%
u = a*z
%% 
% In names, MATLAB distinguishes between lowercase and uppercase letters.
% Variables are not declared but arise by an assignment. The list of
% currently defined variables can be obtained using a command |who|
who
%%
% More details about defined variable prints the function |whos| or the
% window *Workspace* in MATLAB IDE
whos
%%
% Arithmetic in MATLAB is "robust"
infinity = a/0
%%
zero = 1 / infinity
%%
WhatIsThis = infinity / infinity
%%
% 'NaN' is an abbreviation of "Not a Number".
%
% MATLAB can take square roots also form negative numbers
sqrt(-1)
%%
% MATLAB computes naturally with complex numbers. The imaginary unit can be
% denoted as |i| or |j|
(1+2i)*(1-2j)
%%
% MATLAB is a powerful calculator. The value of the last evaluated
% expression not assigned to any variable is stored in the variable |ans|
% 
2^3
%%
2^4
%%
2^100
%%
20^3-3*23/8
%%
sqrt(ans)
%%
% MATLAB knows many arithmetic, goniometric and other functions:
sin(4)
%%
sin(3*pi)+cos(1.23)^2
%%
log(2430)
%%
% What is the base for the logarithm function |log|? The help gives us the
% relevant information
help log
%%
% Better hypertext help can be obtained from the menu behind the button
% with a question mark within a blue circle. Another help with functions is
% provided by the button marked by _fx_ within the left margin of the
% *Command Window* of IDE. Further possibilities are demo-videos and
% demo-examples.
%%
% 
% <<HelpImg.jpg>>
% 
%% Matrix operations
% Standard matrix operations but also *matrix division*!
%
% *Standard Matrix Operations*
%
% An expression followed by the semicolon ';' is evaluated but the result
% is not displayed. Addition, subtraction
A = [1 2 3; 2 3 1; 0 7 3];
B = [1 1 1; 3 2 1; 7 6 5];
C = [3 3; 3 3; 4 3];
whos
%%
A+B
%%
A-B
%%
% The command |A+C| triggers an error message
% 
%  Error using ==> plus Matrix dimensions must agree.
%
% That is why it is not executed in this script.
%
%%
% Matrix multiplication:
D = A*C
%%
A*B
%%
B
%%
% A multiplication by a scalar value:
3*B
%%
% Multiplying only the corresponding components |.*|:
A.*B
%%
% *Matrix transpose*
%
% For a matrix of real values |A|, the expression |A'| returns the
% transpose of |A|.
A = [ 1 2 3; 4 5 6]
%%
A'
%%
% Actually, |A'| is the complex conjugate transpose of |A|, i.e. if |A| is
% complex with non-zero imaginary part, then the matrix |A'| is not only
% transposed but all imaginary parts of all components have sign opposite
% to the corresponding component of |A|.
B = A+1i*(rand(2,3)-0.5)
%%
B'
%%
% In case we would like to obtain 'plain' transpose for a complex matrix,
% we must precede the apostrophe by a dot:
B.'
%%
% *Functions Generating Matrices*
%
% |zeros(m,n)| is a zero matrix of size _mxn_
zeros(2,3)
%%
% |ones(m,n)| is a matrix of size _mxn_ containing ones only.
ones(2,3)
%%
% What vector or matrix we obtain by calling |ones(4)|?
%
% 
% |eye(n)| is the identity matrix of size _nxn_
eye(3)
%%
% |diag(v)| constructs a diagonal matrix which is zero except its diagonal
% that contains values from the vector |v|.
diag([1,2,3])
%%

%%
% *Solving systems of linear equations by "matrix division"*
%
% If a matrix _A_ is not singular, then the system of equations
% 
% $$Ax=b$$
%
% has a solution
%
% $$x=A^{-1}b$$
%
% In MATLAB we use the backslash as an operator of "division from the
% left". At first, we generate a random matrix |A|
A=rand(3,3)
%%
% and a random vector |b|
b=rand(3,1)
%%
% Then we solve the system
%
% $$Ax=b:$$
x=A\b
%%
% We can validate the result (obtained by a numeric method):
A*x-b
%% How to Generate Vectors
% For various purposes, we will need vectors with arithmetic and other
% sequences.
%
% The growing sequence of integers between |A| and |B| can be generated as
% |A:B|
t = 1:6
%%
% This is an arithmetic sequence with the step of size 1 but the size of
% the step can be arbitrary. The sequence from |A| to |B| with step of size
% |Step| we get as |A:Step:B|
t=-1:0.2:1
%%
% Even negative steps are possible
t = 20:-3:1
%%
% Similarly, the function |linspace(A,B,N)| returns |N| numbers which
% divide the interval
% 
% $$<A,B>$$
%
% into |N-1| segments of equal size.
linspace(0,1,11)
%%
% The logarithmic sequence between |10^A| and |10^B| is produced by the
% function |logspace(A,B,N)|
logspace(1,6,7)
%% Vectorized Functions
% Whenever possible, each function in MATLAB is so-called vectorized. This
% means that the argument of the function can be a vector (or a matrix) and
% the function returns another vector (or matrix) of the same size whose
% entries are the function values of the entries of the original argument.
% E.g. the table of the function _sin(2t)_ on the interval
%
% $$ <0 ,\pi>$$
 t = 0:0.1:pi
 %%
 % is obtained as
tab = sin(2*t)
%% 
% We can plot easily a graph of this function
plot(t,tab)
%%
% Be careful with multiplication and division. Usually, we want to compute a
% function for each entry of an array. Example: We want to draw a graph of
% the function
%
% $$1/(1+x^2)$$
x = -2:0.1:2;
y = 1./(1+x.^2);
plot(x,y)
%% Function Plotting
% MATLAB enables us to draw both 2D and 3D graphs. We have already seen
% the plotting of a 2D graph with the function |plot(x,y)|. This function
% accepts also a third parameter. The third parameter specifies the
% parameters of the drawing. E.g. the color of the graph - the red color:
plot(x,y,'r')
%%
% For a green graph with a dashed line:
plot(x,y,'g--')
%%
% Without connecting the depicted points. Points as little circles:
plot(x,y,'o')
%%
% Without connecting the depicted points. Points as little crosses:
plot(x,y,'+')
%%
% For further possibilities see the help for |plot|.
%
% It is possible to label the graph. |xlabel('string')| and
% |ylabel('string')| change the labels of the x- and y-axes in the graph.
plot(x,y)
xlabel('x values');
ylabel('1/(1+x^2)');
%% 
% |grid| adds a rectangular grid to the plot. |hold on| "holds" the current
% graph and enables to add more graphs into the already drawn  one
hold on
grid
plot(x,sin(x))
%%
% |hold off| stops "holding" of the graph.
hold off
%%
% More than one graph can be plotted also by increasing the number of
% parameters in the call to |plot|.
plot(x,y,'r+--',x,sin(x),'go:')
%% Some Miscellaneous Functions 
% The function |max(x)| returns maximal element if |x| is a vector, or a
% vector of maximal elements in each column if |x| is a matrix. Further
% possibilities for calling |max| are described within the help.
max([2, -33.1,14,9])
%%
A
%%
max(A)
%%
% The function |min(x)| works similarly.
%
% |abs(x)| computes absolute value of entries of |x|.
abs([1, -3,-4, 2])
%%
% |size(x)| returns a vector with sizes of |x| in each dimension.
D = randn(2,3)
size(D)
%%
% |length(x)| yields "length" of the array |x|, i.e. |max(size(x))|.
length(D)
%%
% |save fname| saves the values of all currently defined variables into the
% file with the name |fname.mat|
%
%  save 'Tst.mat'
%%
% |save fname var1 var2| saves only the values of the specified variables
% to the file |fname.mat|
%
% |load fname| reads and restores values of all variables from the file
% |fname.mat|.
%
% |quit| ends MATLAB.
%% Working with Elements of Matrices
C = [1 2; 3 4; 5 6]
%%
% Accessing a single entry in a matrix
C(2,1)
%%
% The whole second row of matrix |C|
C(2,:)
%%
% The second and the third row of the matrix |C|
C(2:3,:)
%%
% The first column of matrix |C|
C(:,1)
%%
% Selecting entries from a vector
x=0:2:14
%%
idx = [ 3 1 4];
x(idx)
%%
% Indexing by a vector can be used also for changing entries of a matrix
% (or vector)
x(idx)=[1,2,3]
%% Formatting Output
% By default, MATLAB prints numbers with limited accuracy (5 digits) and
% often it inserts empty lines between lines of output. This can be
% changed. A higher number of displayed digits we get after the following
% command
format long
A
%%
% It is possible to force a format with exponent (floating-point format)
format long e
A
%%
% Another combination is short format with forced exponent
format short e
A
%%
% Or the short format without forced floating point format. Nevertheless,
% if necessary, the floating point format will be used, otherwise the fixed
% point format is used
format short
A
%%
% One of the most useful settings is the following command which suppresses
% unnecessary empty lines in output by which the out becomes more
% "compact".
format compact
A
%% Further Commands and Options 
% will be shown during the next lab
% 
% * programming with MATLAB,
% * sparse matrices,
% * structures,
% * cell arrays.
% 





