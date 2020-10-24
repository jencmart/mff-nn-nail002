function M = c_02_tema_2_matice(m,n,a,b)
% GENMAT generuje nahodnú maticu čísel.
%          GenMat(m,n) vráti maticu rozmerov mxn s náhodnými číslami z
%          intervalu (0,1).
%          GenMat(m,n,a,b) vráti maticu rozmerov mxn s náhodnými číslami z
%          intervalu (a,b).
if nargin < 3                 
    a=0;                           
    b=1;
end
if b<a
    error('Horna hranica intervalu musi byt vyssia nez dolna')
end
M = rand(m,n) * (b-a) + a;
end

%%
% Všetky premenné vo funkcii sú lokálne. Na globálnu premennú |X| sa však dá
% dostať deklaráciou
%
%  global X
% 
% Prvý blok komentárov (komentár začína znakom percenta) za hlavičkou
% funkcie je vypísaný pri zavolaní nápovedy
%
%  help GenMat
%
% Príklad funkcie viz "A Practical Introduction to Matlab".
% Funkcia sa volá tak ako zabudované funkcie MATLABu. Keď MATLAB zistí, že
% voláte funkciu, tak hľadá súbor s deklaráciou funkcie v adresároch podľa
% ich poradia v zozname |path|. Príkaz
%
%  path('mydir',path)
%
% pridá adresár mydir na začiatok tohoto zoznamu. Príkaz
%
%  path(path,'mydir')
%
% pridá adresár mydir na koniec tohoto zoznamu. Interaktívne je možné
% premennú |path| zmeniť z IDE MATLABu v záložke |HOME| cez menu |Set
% Path|.