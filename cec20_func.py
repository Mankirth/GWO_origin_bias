function [f,g,h] = cec20_func(x,prob_k)
% cec20_func Constrained Optimization Test Suite 
% Abhishek Kumar (email: abhishek.kumar.eee13@iitbhu.ac.in, Indian Institute of Technology (BHU), Varanasi) 

% x -----> ps X D where 'ps': number of population and 'D': Dimension of
% the problem.
% f -----> Objective Function Value.
% g -----> Inequality Consstraints Value; ps X ng where 'ng': number of
% inequality constraints.
% h -----> Equality Constraints Value; ps X nh where 'nh': number of
% equality constraints.
% prob_k -> Index of problem.


[ps,D]=size(x);
global initial_flag
persistent G B P Q L

%% Industrial Chemical Processes		
if(prob_k == 1)
    %% Heat Exchanger Network Design (case 1)
    f = 35.*x(:,1).^0.6 + 35.*x(:,2).^0.6;
    g = zeros(ps,1);
    h(:,1) = 200.*x(:,1).*x(:,4)-x(:,3);
    h(:,2) = 200.*x(:,2).*x(:,6)-x(:,5);
    h(:,3) = x(:,3) - 10000.*(x(:,7)-100);
    h(:,4) = x(:,5) - 10000.*(300-x(:,7));
    h(:,5) = x(:,3) - 10000.*(600-x(:,8));
    h(:,6) = x(:,5) - 10000.*(900-x(:,9));
    h(:,7) = x(:,4).*log(abs(x(:,8)-100)+1e-8)-x(:,4).*log((600-x(:,7))+1e-8)-x(:,8)+x(:,7)+500;
    h(:,8) = x(:,6).*log(abs(x(:,9)-x(:,7))+1e-8)-x(:,6).*log(600)-x(:,9)+x(:,7)+600;
end


if(prob_k == 2)
    %% Heat Exchanger Network Design (case 2)
    f = (x(:,1)./(120*x(:,4))).^0.6+(x(:,2)./(80*x(:,5))).^0.6+(x(:,3)./(40*x(:,6))).^0.6;
    g = zeros(ps,1);
    h(:,1) = x(:,1)-1e4.*(x(:,7)-100);
    h(:,2) = x(:,2)-1e4.*(x(:,8)-x(:,7));
    h(:,3) = x(:,3)-1e4.*(500-x(:,8));
    h(:,4) = x(:,1)-1e4.*(300-x(:,9));
    h(:,5) = x(:,2)-1e4.*(400-x(:,10));
    h(:,6) = x(:,3)-1e4.*(600-x(:,11));
    h(:,7) = x(:,4).*log(abs(x(:,9)-100)+1e-8)-x(:,4).*log(300-x(:,7)+1e-8)-x(:,9)-x(:,7)+400;
    h(:,8) = x(:,5).*log(abs(x(:,10)-x(:,7))+1e-8)-x(:,5).*log(abs(400-x(:,8))+1e-8)-x(:,10)+x(:,7)-x(:,8)+400;
    h(:,9) = x(:,6).*log(abs(x(:,11)-x(:,8))+1e-8)-x(:,6).*log(100)-x(:,11)+x(:,8)+100;
end

if (prob_k == 3)
    %% Optimal Operation of Alkylation Unit
      f = -1.715.*x(:,1)-0.035.*x(:,1).*x(:,6)-4.0565.*x(:,3)-10.0.*x(:,2)+0.063.*x(:,3).*x(:,5);
      h = zeros(ps,1);
      g(:,1) = 0.0059553571.*x(:,6).^2.*x(:,1)+0.88392857.*x(:,3)-0.1175625.*x(:,6).*x(:,1)-x(:,1);
      g(:,2) = 1.1088.*x(:,1)+0.1303533.*x(:,1).*x(:,6)-0.0066033.*x(:,1).*x(:,6).^2-x(:,3);
      g(:,3) = 6.66173269.*x(:,6).^2+172.39878.*x(:,5)-56.596669.*x(:,4)-191.20592.*x(:,6)-10000;
      g(:,4) = 1.08702.*x(:,6)+0.32175.*x(:,4)-0.03762.*x(:,6).^2-x(:,5)+56.85075;
      g(:,5) = 0.006198.*x(:,7).*x(:,4).*x(:,3)+2462.3121.*x(:,2)-25.125634.*x(:,2).*x(:,4)-x(:,3).*x(:,4);
      g(:,6) = 161.18996.*x(:,3).*x(:,4)+5000.0.*x(:,2).*x(:,4)-489510.0.*x(:,2)-x(:,3).*x(:,4).*x(:,7);
      g(:,7) = 0.33.*x(:,7)-x(:,5)+44.333333;
      g(:,8) = 0.022556.*x(:,5)-0.007595.*x(:,7)-1.0;
      g(:,9) = 0.00061.*x(:,3)-0.0005.*x(:,1)-1.0;
      g(:,10)= 0.819672.*x(:,1)-x(:,3)+0.819672;
      g(:,11)= 24500.0.*x(:,2)-250.0.*x(:,2).*x(:,4)-x(:,3).*x(:,4);
      g(:,12)= 1020.4082.*x(:,4).*x(:,2)+1.2244898.*x(:,3).*x(:,4)-100000.*x(:,2);
      g(:,13)= 6.25.*x(:,1).*x(:,6)+6.25.*x(:,1)-7.625.*x(:,3)-100000;
      g(:,14)= 1.22.*x(:,3)-x(:,6).*x(:,1)-x(:,1)+1.0;
end

if (prob_k == 4)
    %% Reactor Network Design (RND)
    k1 = 0.09755988;
    k2 = 0.99.*k1;
    k3 = 0.0391908;
    k4 = 0.9.*k3;
    f = -x(:,4);
    h(:,1) = x(:,1)+k1.*x(:,2).*x(:,5)-1;
    h(:,2) = x(:,2)-x(:,1)+k2.*x(:,2).*x(:,6);
    h(:,3) = x(:,3)+x(:,1)+k3.*x(:,3).*x(:,5)-1;
    h(:,4) = x(:,4)-x(:,3)+x(:,2)-x(:,1)+k4.*x(:,4).*x(:,6);
    g(:,1) = x(:,5).^0.5+x(:,6).^0.5-4;
end

if(prob_k == 5)
    %% Haverly's Pooling Problem
    f = -(9.*x(:,1)+15.*x(:,2)-6.*x(:,3)-16.*x(:,4)-10.*(x(:,5)+x(:,6)));
    g(:,1) = x(:,9).*x(:,7)+2.*x(:,5)-2.5.*x(:,1);
    g(:,2) = x(:,9).*x(:,8)+2.*x(:,6)-1.5.*x(:,2);
    h(:,1) = x(:,7)+x(:,8)-x(:,3)-x(:,4);
    h(:,2) = x(:,1)-x(:,7)-x(:,5);
    h(:,3) = x(:,2)-x(:,8)-x(:,6);
    h(:,4) = x(:,9).*x(:,7)+x(:,9).*x(:,8)-3.*x(:,3)-x(:,4);
end

%% Power Electronic Problems		

if (prob_k == 45)
    %% SOPWM for 3-level Invereters
    m = 0.32;
    s = (-ones(1,25)).^(2:26);
    k = [5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97];
    for i = 1:ps
        su = 0;
        for j = 1:31
            su2 = 0;
            for l = 1:D
                su2 = su2 + s(l).*cos(k(j).*x(i,l)*pi/180);
            end
            su = su + su2.^2./k(j).^4;
        end
        f(i,1) = (su).^0.5./(sum(1./k.^4)).^0.5;
    end
    g = zeros(ps,D-1);
    for i = 1:D-1
        g(:,i) = x(:,i)-x(:,i+1)+1e-6;
    end
    h = sum(s.*cos(x*pi/180),2)-m;

end

if (prob_k == 46)
    %% SOPWM for 5-level Inverters
    m = 0.32;
    s = [1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1];
    k = [5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97];
    for i = 1:ps
        su = 0;
        for j = 1:31
            su2 = 0;
            for l = 1:D
                su2 = su2 + s(l).*cos(k(j).*x(i,l)*pi/180);
            end
            su = su + su2.^2./k(j).^4;
        end
        f(i,1) = 0.5.*(su).^0.5./(sum(1./k.^4)).^0.5;
    end
    g = zeros(ps,D-1);
    for i = 1:D-1
        g(:,i) = x(:,i)-x(:,i+1)+1e-6;
    end
    h = sum(s.*cos(x*pi/180),2)-2*m;
end

if (prob_k == 47)
    %% SOPWM for 7-level Inverters
    m = 0.36;
    s = [1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,1,1,1];
    k = [5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97];
    for i = 1:ps
        su = 0;
        for j = 1:31
            su2 = 0;
            for l = 1:D
                su2 = su2 + s(l).*cos(k(j).*x(i,l)*pi/180);
            end
            su = su + su2.^2./k(j).^4;
        end
        f(i,1) = 1/3.*(su).^0.5./(sum(1./k.^4)).^0.5;
    end
    g = zeros(ps,D-1);
    for i = 1:D-1
        g(:,i) = x(:,i)-x(:,i+1)+1e-6;
    end
    h = sum(s.*cos(x*pi/180),2)-3*m;    
end

if (prob_k == 48)
    %% SOPWM for 9-level Inverters
    m = 0.32;
    s = [1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,1];
    k = [5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97];
    for i = 1:ps
        su = 0;
        for j = 1:31
            su2 = 0;
            for l = 1:D
                su2 = su2 + s(l).*cos(k(j).*x(i,l)*pi/180);
            end
            su = su + su2.^2./k(j).^4;
        end
        f(i,1) = 1/4.*(su).^0.5./(sum(1./k.^4)).^0.5;
    end
    g = zeros(ps,D-1);
    for i = 1:D-1
        g(:,i) = x(:,i)-x(:,i+1)+1e-6;
    end
    h = sum(s.*cos(x*pi/180),2)-4*m; 
end

if (prob_k == 49)
    %% SOPWM for 11-level Inverters
    m = 0.3333;
    s = [1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,1,1,-1,-1,1,-1,-1];
    k = [5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97];
    for i = 1:ps
        su = 0;
        for j = 1:31
            su2 = 0;
            for l = 1:D
                su2 = su2 + s(l).*cos(k(j).*x(i,l)*pi/180);
            end
            su = su + su2.^2./k(j).^4;
        end
        f(i,1) = 1/5.*(su).^0.5./(sum(1./k.^4)).^0.5;
    end
    g = zeros(ps,D-1);
    for i = 1:D-1
        g(:,i) = x(:,i)-x(:,i+1)+1e-6;
    end
    h = sum(s.*cos(x*pi/180),2)-5*m; 
end

if (prob_k == 50)
    %% SOPWM for 13-level Inverters
    m = 0.32;
    s = [1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1];
    k = [5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97];
    for i = 1:ps
        su = 0;
        for j = 1:31
            su2 = 0;
            for l = 1:D
                su2 = su2 + s(l).*cos(k(j).*x(i,l)*pi/180);
        
            su = su + su2.^2./k(j).^4;
    
        f(i,1) = 1/6.*(su).^0.5./(sum(1./k.^4)).^0.5;

    g = zeros(ps,D-1);
    for i = 1:D-1
        g(:,i) = x(:,i)-x(:,i+1)+1e-6;

    h = sum(s.*cos(x*pi/180),2)-6*m; 

g=g';
h=h';

end

% Program to for Admittance And Impedance Bus Formation....

function Y = ybus(linedata,f)  % Returns Y
linedata(:,4) = linedata(:,4).*f;
% linedata(:,3:4) = linedata(:,3:4).*10000/127^2;
% linedata(:,3:4) = linedata(:,3:4);


fb = linedata(:,1);             % From bus number...
tb = linedata(:,2);             % To bus number...
r = linedata(:,3);              % Resistance, R...
x = linedata(:,4);              % Reactance, X...
b = linedata(:,5);              % Ground Admittance, B/2...
a = linedata(:,6);              % Tap setting value..
z = r + i*x;                    % z matrix...
y = 1./z;                       % To get inverse of each element...
b = i*b;                        % Make B imaginary...

nb = max(max(fb),max(tb));      % No. of buses...
nl = length(fb);                % No. of branches...
Y = zeros(nb,nb);               % Initialise YBus...
 
 % Formation of the Off Diagonal Elements...
 for k = 1:nl
     Y(fb(k),tb(k)) = Y(fb(k),tb(k)) - y(k)/a(k);
     Y(tb(k),fb(k)) = Y(fb(k),tb(k));
 end
 
 % Formation of Diagonal Elements....
 for m = 1:nb
     for n = 1:nl
         if fb(n) == m
             Y(m,m) = Y(m,m) + y(n)/(a(n)^2) + b(n);
         elseif tb(n) == m
             Y(m,m) = Y(m,m) + y(n) + b(n);
         end
     end
 end
end
function ff = OBJ11(x,n)
a = x(1); b = x(2); c = x(3); e = x(4); f = x(5); l = x(6); 
 Zmax = 99.9999; P = 100;
if n == 1
     fhd = @(z) P.*b.*sin(acos((a.^2+(l-z).^2+e.^2-b.^2)./(2.*a.*sqrt((l-z).^2+e.^2)))+acos((b.^2+(l-z).^2+e.^2-a.^2)./(2.*b.*sqrt((l-z).^2+e.^2))))./....
       (2.*c.*cos(acos((a.^2+(l-z).^2+e.^2-b.^2)./(2.*a.*sqrt((l-z).^2+e.^2)))+atan(e./(l-z))));
else
    fhd = @(z) -(P.*b.*sin(acos((a.^2+(l-z).^2+e.^2-b.^2)./(2.*a.*sqrt((l-z).^2+e.^2)))+acos((b.^2+(l-z).^2+e.^2-a.^2)./(2.*b.*sqrt((l-z).^2+e.^2))))./....
       (2.*c.*cos(acos((a.^2+(l-z).^2+e.^2-b.^2)./(2.*a.*sqrt((l-z).^2+e.^2)))+atan(e./(l-z)))));
end
options = optimset('Display','off');
 [~,ff]= fminbnd(fhd,0,Zmax,options); 
end

function [Weight] = function_fitness(section)

E   = 6.98*1e10;      % Young's elastic modulus (N/m^2)
A   = section;        % area of bar (m^2)
rho = 2770;           % density of material (kg/m^3)
%--------------------------------------------------------------------------
%           1         2       3       4       5     6                     
gcoord = [18.288,  18.288,  9.144,  9.144,      0,  0 
           9.144,       0,  9.144,      0,  9.144,  0];
%          1  2  3  4  5  6  7  8  9  10
element = [3, 1, 4, 2, 3, 1, 4, 3, 2, 1
           5, 3, 6, 4, 4, 2, 5, 6, 3, 4];
%--------------------------------------------------------------------------
% calculate Weight matrix
Weight = 0;
for i=1:length(element)
    nd = element(:,i);
    x  = gcoord(1,nd); y = gcoord(2,nd);
    % compute long of each bar
    le = sqrt((x(2)-x(1))^2 + (y(2)-y(1))^2);
    Weight =  Weight + rho*le*A(i);
end
end
function [c,ceq] = ConsBar10(x)
type = '2D';
E    = 6.98*1e10;      % Young's elastic modulus (N/m^2)
A    = x;
rho  = 2770;           % density of material (kg/m^3)
%--------------------------------------------------------------------------
%           1        2        3       4       5     6          
gcoord  = [18.288,  18.288,  9.144,  9.144,      0,  0 
           9.144,       0,  9.144,      0,  9.144,  0];
%          1  2  3  4  5  6  7  8  9  10
element = [3, 1, 4, 2, 3, 1, 4, 3, 2, 1
           5, 3, 6, 4, 4, 2, 5, 6, 3, 4];
nel     = length(element);    % total element
nnode   = length(gcoord);     % total node
ndof    = 2;                  % number of degree of freedom of one node
sdof    = nnode*ndof;         % total dgree of freedom of system
% plotModel( type,gcoord,element );
% calculate stiffness matrix 
[ K,M ] = Cal_K_and_M( type,gcoord,element,A,rho,E );
% add non-structural mass
addedMass = 454; %kg
for idof = 1:sdof
    M(idof,idof) = M(idof,idof) + addedMass;
end
% apply boundary
bcdof   = [(5:6)*2-1, (5:6)*2];     % boundary condition displacement
% Giai phuong trinh tim tri rieng va vector rieng
[omega_2,~]=eigens(K,M,bcdof); 
f=sqrt(omega_2)/2/pi;
% f(1:5)
c1 = 7/f(1) -1;
c2 = 15/f(2)-1;
c3 = 20/f(3)-1;
c = [c1,c2,c3];
ceq = [];
end

function [ K,M ] = Cal_K_and_M( type,gcoord,element,A,rho,E )
% calculate K and M
nel     = length(element);    % total element
nnode   = length(gcoord);     % total node
switch type
    case '3D'
        ndof    = 3;                  % number of degree of freedom of one node
        sdof    = nnode*ndof;         % total dgree of freedom of system
        K       = zeros(sdof,sdof);
        M       = zeros(sdof,sdof);
        for iel=1:nel
            nd = element(:,iel);
            x  = gcoord(1,nd); y = gcoord(2,nd); z = gcoord(3,nd);
            % compute long of each bar
            le = sqrt((x(2)-x(1))^2 + (y(2)-y(1))^2 + (z(2)-z(1))^2);
            % compute direction cosin
            l_ij = (x(2)-x(1))/le;      % Eq.8.19
            m_ij = (y(2)-y(1))/le;      % Eq.8.19
            n_ij = (z(2)-z(1))/le;      % Eq.8.19
            % compute transform matrix
            Te = [l_ij m_ij  n_ij   0       0     0;
                0    0      0   l_ij   m_ij   n_ij];
            % compute stiffness matrix of element
            ke = A(iel)*E/le*[1 -1; -1  1];
            ke = Te'*ke*Te;
            me = rho*le*A(iel)*[2 0 0 1 0 0
                0 2 0 0 1 0;
                0 0 2 0 0 1;
                1 0 0 2 0 0;
                0 1 0 0 2 0;
                0 0 1 0 0 2]/6;
            % find index assemble
            index   = [3*nd(1)-2 3*nd(1)-1 3*nd(1)  3*nd(2)-2 3*nd(2)-1  3*nd(2)];
            % assemble ke in K
            K(index,index) = K(index,index) + ke;
            M(index,index) = M(index,index) + me;
        end
        
    case '2D'
        ndof    = 2;                  % number of degree of freedom of one node
        sdof    = nnode*ndof;         % total dgree of freedom of system
        K       = zeros(sdof,sdof);
        M       = zeros(sdof,sdof);
        for iel=1:nel
            nd = element(:,iel);
            x  = gcoord(1,nd); y = gcoord(2,nd);
            % compute long of each bar
            le = sqrt((x(2)-x(1))^2 + (y(2)-y(1))^2);
            % compute direction cosin
            l_ij = (x(2)-x(1))/le;
            m_ij = (y(2)-y(1))/le;
            % compute transform matrix
            Te = [l_ij m_ij   0      0 ;
                0    0   l_ij   m_ij];
            
            % compute stiffness matrix of element
            ke = A(iel)*E/le*[1 -1;
                -1  1];
            ke = Te'*ke*Te;
            me = rho*le*A(iel)*[2 0 1 0;
                0 2 0 1
                1 0 2 0
                0 1 0 2]/6; % lumped mass matrix
            % find index assemble
            index   = [2*nd(1)-1 2*nd(1)  2*nd(2)-1  2*nd(2)];
            % assemble ke in K
            K(index,index) = K(index,index) + ke;
            % assemble me in M
            M(index,index) = M(index,index) + me;
        end
end
end

function [L,X]=eigens(K,M,b)
  [nd,nd]=size(K);
  fdof=[1:nd]';
%
  if nargin==3
    pdof=b(:);
    fdof(pdof)=[]; 
    if nargout==2
      [X1,D]=eig(K(fdof,fdof),M(fdof,fdof));
      [nfdof,nfdof]=size(X1);
      for j=1:nfdof;
        mnorm=sqrt(X1(:,j)'*M(fdof,fdof)*X1(:,j));
        X1(:,j)=X1(:,j)/mnorm;
      end
      d=diag(D);
      [L,i]=sort(d);
      X2=X1(:,i);
      X=zeros(nd,nfdof);
      X(fdof,:)=X2;
    else
      d=eig(K(fdof,fdof),M(fdof,fdof));
      L=sort(d);
    end
  else
    if nargout==2
      [X1,D]=eig(K,M);
      for j=1:nd;
        mnorm=sqrt(X1(:,j)'*M*X1(:,j));
        X1(:,j)=X1(:,j)/mnorm;
      end
      d=diag(D);
      [L,i]=sort(d);
      X=X1(:,i);
    else
      d=eig(K,M);
      L=sort(d);
    end
  end
end


function all_power = Fitness(interval_num, interval, fre, N, coordinate, ...,
            a, kappa, R, k, c, cut_in_speed, rated_speed, cut_out_speed, evaluate_method)
all_power = 0;                 
for i = 1 : interval_num
   interval_dir = (i - 0.5) * interval;
   [power_eva] = eva_power(i, interval_dir, N, coordinate, ...,
            a, kappa, R,k(i), c(i), cut_in_speed, rated_speed, cut_out_speed, evaluate_method);
    all_power = all_power + fre(i) * sum(power_eva);
end
end

function power_eva = eva_power(interval_dir_num, interval_dir, N, coordinate, ...,
           a, kappa, R, k, c, cut_in_speed, rated_speed, cut_out_speed, evaluate_method)

if(strcmp(evaluate_method, 'caching'))
    [vel_def] = eva_func_deficit_caching(interval_dir_num ,N, coordinate, interval_dir, a, kappa, R);
else
    [vel_def] = eva_func_deficit(interval_dir_num, N, coordinate, interval_dir, a, kappa, R);
end
interval_c(1 : N) = 0;
for i = 1 : N
   interval_c(i) = c * (1 - vel_def(i)); 
end
n_ws = (rated_speed - cut_in_speed) / 0.3;
power_eva(1 : N) = 0;
for i = 1 : N
    for j = 1 : n_ws
        v_j_1 = cut_in_speed + (j - 1) * 0.3;
        v_j = cut_in_speed + j * 0.3;
        power_eva(i) = power_eva(i) + 1500 * exp((v_j_1 + v_j) / 2 - 7.5) ./ (5 + exp((v_j_1 + v_j) / 2 - 7.5)) * ...,
            (exp(-(v_j_1 / interval_c(i))^k) - exp(-(v_j / interval_c(i))^k));
    end
    power_eva(i) = power_eva(i) + 1500 * (exp(-(rated_speed / interval_c(i))^k) - exp(-(cut_out_speed / interval_c(i))^k));
end
end


function[vel_def] = eva_func_deficit(interval_dir_num, N, coordinate, theta, a, kappa, R)


global thetaVeldefijMatrix;

vel_def(1 : N) = 0;

for i = 1 : N
    vel_def_i = 0;
    for j = 1 : N   
        [affected, dij] = downstream_wind_turbine_is_affected(coordinate, j, i, theta, kappa, R);
        if(affected)  
            def = a / (1 + kappa * dij / R)^2;
%             def = restrict(def, 1);
            thetaVeldefijMatrix(i, j, interval_dir_num) = def;
            vel_def_i = vel_def_i + def^2;  
        else
            thetaVeldefijMatrix(i, j, interval_dir_num) = 0;
        end  
    end
%     vel_def_i = restrict(vel_def_i, 1);
    vel_def(i) = sqrt(vel_def_i);
end
end

function[vel_def] = eva_func_deficit_caching(interval_dir_num, N, coordinate, theta, a, kappa, R)

global thetaVeldefijMatrix;
global turbineMoved;

vel_def(1 : N) = 0;
movedTurbine = 1;
for i = 1 : N
    if(turbineMoved(i) == 1)
        movedTurbine = i;
    end
end

for i = 1 : N

    vel_def_i = 0;
  
    if(i ~= movedTurbine)
        [affected, dij] = downstream_wind_turbine_is_affected(coordinate, movedTurbine, i, theta, kappa, R);
        if(affected)  
            def = a / (1 + kappa * dij / R)^2;
            def = restrict(def, 1);
        else      
            def = 0;
        end 
        vel_def_i = sum((thetaVeldefijMatrix(i, :, interval_dir_num)).^2) - (thetaVeldefijMatrix(i, movedTurbine, interval_dir_num))^2 + def^2;
        thetaVeldefijMatrix(i, movedTurbine, interval_dir_num) = def;
    else
        for j = 1 : N   
            [affected, dij] = downstream_wind_turbine_is_affected(coordinate, j, i, theta, kappa, R);
            if(affected)  
                def = a / (1 + kappa * dij / R)^2;
                def = restrict(def, 1);
            else
                def = 0;      
            end
            vel_def_i = vel_def_i + def^2; 
            thetaVeldefijMatrix(i,j,interval_dir_num) = def;
        end
    end
    vel_def_i = restrict(vel_def_i, 1);
    vel_def(i) = sqrt(vel_def_i);
end
end

function[affected, dij] = downstream_wind_turbine_is_affected(coordinate, upstream_wind_turbine, ...,
    downstream_wind_turbine, theta, kappa, R)

    affected = 0;
    Tijx = (coordinate(2 * downstream_wind_turbine - 1) - coordinate(2 * upstream_wind_turbine - 1));
    Tijy = (coordinate(2 * downstream_wind_turbine) - coordinate(2 * upstream_wind_turbine));
    dij = cosd(theta) * Tijx + sind(theta) * Tijy;
    lij = sqrt((Tijx^2 + Tijy^2) - (dij)^2);
    l = dij * kappa + R;
    if((upstream_wind_turbine ~= downstream_wind_turbine) && (l > lij-R) && (dij > 0))
        affected = 1;
    end
end



%%%%%%%%%% MESH-INDEPENDENCY FILTER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dcn]=check(nelx,nely,rmin,x,dc)
dcn=zeros(nely,nelx);
for i = 1:nelx
  for j = 1:nely
    sum=0.0; 
    for k = max(i-floor(rmin),1):min(i+floor(rmin),nelx)
      for l = max(j-floor(rmin),1):min(j+floor(rmin),nely)
        fac = rmin-sqrt((i-k)^2+(j-l)^2);
        sum = sum+max(0,fac);
        dcn(j,i) = dcn(j,i) + max(0,fac)*x(l,k)*dc(l,k);
      end
    end
    dcn(j,i) = dcn(j,i)/(x(j,i)*sum);
  end
end
end
%%%%%%%%%% FE-ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U]=FE(nelx,nely,x,penal)
[KE] = lk; 
K = sparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1));
F = sparse(2*(nely+1)*(nelx+1),1); U = zeros(2*(nely+1)*(nelx+1),1);
for elx = 1:nelx
  for ely = 1:nely
    n1 = (nely+1)*(elx-1)+ely; 
    n2 = (nely+1)* elx   +ely;
    edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
    K(edof,edof) = K(edof,edof) + x(ely,elx)^penal*KE;
  end
end
% DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
%F(2,1) = -10000;
F(2*(nely+1)*(nelx+1),1)=-10000; 
%fixeddofs   = union([1:2:2*(nely+1)],[2*(nelx+1)*(nely+1)]);
fixeddofs   = [1:2*(nely+1)];
alldofs     = [1:2*(nely+1)*(nelx+1)];
freedofs    = setdiff(alldofs,fixeddofs);
% SOLVING
U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);      
U(fixeddofs,:)= 0;
end
%%%%%%%%%% ELEMENT STIFFNESS MATRIX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [KE]=lk
E = 206000000.; 
nu = 0.3;
k=[ 1/2-nu/6   1/8+nu/8 -1/4-nu/12 -1/8+3*nu/8 ... 
   -1/4+nu/12 -1/8-nu/8  nu/6       1/8-3*nu/8];
KE = E/(1-nu^2)*[ k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8)
                  k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3)
                  k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2)
                  k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5)
                  k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4)
                  k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7)
                  k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6)
                  k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)];
end

