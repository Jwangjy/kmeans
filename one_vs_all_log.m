%% Advanced Econometrics HW4 - Jieyang Wang
%% Q1

close all
clear all
clc

load iris
load Setosa_thetaval
load Versicolor_thetaval
load Virginica_thetaval

Xt1 = meas(1:50, 1:4);
Xt2 = meas(51:100, 1:4);
Xt3 = meas(101:150, 1:4);
Xt = [Xt1; Xt2; Xt3];
[m n] = size(Xt);
Xtest = zeros(m,2);
for i = 1:m
    Xtest(i,:) = [mean(Xt(i,:)) var(Xt(i,:))];
end 

Xtest = [ones(m, 1) Xtest];

y1 = meas(1:50, 5);
y2 = meas(51:100, 5);
y3 = meas(101:150, 5);
y = [y1; y2; y3];

plotClass(Xtest(:,2:3)',y');

figure(1);
hold on

plot_x = [min(Xtest(:,2))-2,  max(Xtest(:,2))+2];

% Calculate the decision boundary line
plot_y1 = (-1./theta1(3)).*(theta1(2).*plot_x + theta1(1));

% Plot, and adjust axes for better viewing
plot(plot_x, plot_y1,'b')

hold on

% Calculate the decision boundary line
plot_y2 = (-1./theta2(3)).*(theta2(2).*plot_x + theta2(1));

% Plot, and adjust axes for better viewing
plot(plot_x, plot_y2,'r')

hold on

% Calculate the decision boundary line
plot_y3 = (-1./theta3(3)).*(theta3(2).*plot_x + theta3(1));

% Plot, and adjust axes for better viewing
plot(plot_x, plot_y3,'g')

hold off
grid off
legend('Setosa','Versicolor','Virginica')
axis([0 5 2 6]);

%%
prediction1 = 1./(1+exp(-Xtest*theta1));
prediction2 = 1./(1+exp(-Xtest*theta2));
prediction3 = 1./(1+exp(-Xtest*theta3));

prediction = zeros(m,1);
for i = 1:m
    if prediction1(i,:)>=prediction2(i,:) && prediction1(i,:)>=prediction3(i,:)
        prediction(i,:) = 1;
    elseif prediction2(i,:)>=prediction1(i,:) && prediction2(i,:)>=prediction3(i,:)
        prediction(i,:) = 2;
    else
        prediction(i,:) = 3;
    end
end

% Insert here your computations of the confusion matrix, precisions, recalls, F tests and accuracy

figure(2);
C = confusionchart(y,prediction);
disp('Confusion matrix is:'); disp(C)
C.RowSummary = 'row-normalized';
C.ColumnSummary = 'column-normalized';

%%
p1 = 1;
r1 = 1;
p2 = 0.84;
r2 = 0.913;
p3 = 0.92;
r3 = 0.852;
f1 = 2*(p1*r1)/(p1+r1);
f2 = 2*(p2*r2)/(p2+r2);
f3 = 2*(p3*r3)/(p3+r3);
disp('Precision and Recall for class 1 are:'); disp(p1); disp(r1);
disp('Precision and Recall for class 2 are:'); disp(p2); disp(r2);
disp('Precision and Recall for class 3 are:'); disp(p3); disp(r3);
disp('F measure for class 1 is:'); disp(f1);
disp('F measure for class 2 is:'); disp(f2);
disp('F measure for class 3 is:'); disp(f3);

Acc = (50+42+46)/150;
disp('Accuracy of the system is:');disp(Acc);


%% Q2a)

clear all
clc

rng(2021);
load iris
X = meas(:,1:4);
clust = zeros(size(X,1),6);
for i=1:6
clust(:,i) = kmeans(X,i,'emptyaction','singleton',...
        'replicate',5);
end
va = evalclusters(X,clust,'CalinskiHarabasz')

% Optimal k is 3

k = 3;
[id,C] = kmeans(X,k,'emptyaction','singleton',...
    'replicate',5);
% Reordering clusters to match previous flower numbering
idx = zeros(150:1);
for i = 1:150
    if id(i)==2
        idx(i)=1;
    elseif id(i)==1
        idx(i)=3;
    else
        idx(i)=2;
    end
end

figure(3);
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','SE')
title 'Cluster Assignments and Centroids'
hold off

%%
y = meas(:,5);

figure(4);
C = confusionchart(y,idx);
disp('Confusion matrix is:'); disp(C)
C.RowSummary = 'row-normalized';
C.ColumnSummary = 'column-normalized';

%%
p1 = 1;
r1 = 1;
p2 = 0.96;
r2 = 0.774;
p3 = 0.72;
r3 = 0.947;
f1 = 2*(p1*r1)/(p1+r1);
f2 = 2*(p2*r2)/(p2+r2);
f3 = 2*(p3*r3)/(p3+r3);
disp('Precision and Recall for class 1 are:'); disp(p1); disp(r1);
disp('Precision and Recall for class 2 are:'); disp(p2); disp(r2);
disp('Precision and Recall for class 3 are:'); disp(p3); disp(r3);
disp('F measure for class 1 is:'); disp(f1);
disp('F measure for class 2 is:'); disp(f2);
disp('F measure for class 3 is:'); disp(f3);

Acc = (50+48+36)/150;
disp('Accuracy of the system is:');disp(Acc);

%% Q2b)

rng(2021);
Xn = normalize(X);
[id,C] = kmeans(Xn,k,'emptyaction','singleton',...
    'replicate',5);
% Reordering clusters to match previous flower numbering
rng(2021);
idx = zeros(150:1);
for i = 1:150
    if id(i)==1
        idx(i)=1;
    elseif id(i)==2
        idx(i)=3;
    else
        idx(i)=2;
    end
end

figure(5);
plot(Xn(idx==1,1),Xn(idx==1,2),'r.','MarkerSize',12)
hold on
plot(Xn(idx==2,1),Xn(idx==2,2),'b.','MarkerSize',12)
plot(Xn(idx==3,1),Xn(idx==3,2),'g.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','SE')
title 'Cluster Assignments and Centroids'
hold off

%%
figure(6);
C = confusionchart(y,idx);
disp('Confusion matrix is:'); disp(C)
C.RowSummary = 'row-normalized';
C.ColumnSummary = 'column-normalized';

%%
p1 = 1;
r1 = 1;
p2 = 0.78;
r2 = 0.736;
p3 = 0.72;
r3 = 0.766;
f1 = 2*(p1*r1)/(p1+r1);
f2 = 2*(p2*r2)/(p2+r2);
f3 = 2*(p3*r3)/(p3+r3);
disp('Precision and Recall for class 1 are:'); disp(p1); disp(r1);
disp('Precision and Recall for class 2 are:'); disp(p2); disp(r2);
disp('Precision and Recall for class 3 are:'); disp(p3); disp(r3);
disp('F measure for class 1 is:'); disp(f1);
disp('F measure for class 2 is:'); disp(f2);
disp('F measure for class 3 is:'); disp(f3);

Acc = (50+39+36)/150;
disp('Accuracy of the system is:');disp(Acc);

% The accuracy of the system has deproved after normalization.

%% Q3a)

clear all
clc

rng(2021);
load iris;
X = meas(:,1:4);
[U,S,V]=svd(X);
s = sum(S);
Var1 = s(1)/sum(s);
Var2 = s(2)/sum(s);
Var3 = s(3)/sum(s);
Var4 = s(4)/sum(s);
Var12 = (s(1)+s(2))/sum(s);
Var123 = (s(1)+s(2)+s(3))/sum(s);
disp('Variance explained by principal component 1 is'); disp(Var1);
disp('Variance explained by principal component 2 is'); disp(Var2);
disp('Variance explained by principal component 3 is'); disp(Var3);
disp('Variance explained by principal component 4 is'); disp(Var4);
disp('Variance explained by principal component 1,2 is'); disp(Var12);
disp('Variance explained by principal component 1,2,3 is'); disp(Var123);

%% Q3b) - Real line of 1 principal component
Ureduce1 = U(:,1:1);
Sreduce1 = S(1,1);
z1 = Ureduce1.*Sreduce1;

data1 = z1(1:50);
data2 = z1(51:100);
data3 = z1(101:150);

figure(7);
hAxes = axes('NextPlot','add',...           
             'DataAspectRatio',[1 1 1],...  
             'XLim',[min(z1)-2 max(z1)+2],... 
             'YLim',[0 eps],...             
             'Color','none');               
plot(data1,0,'r.');  
plot(data2,0,'b.'); 
plot(data3,0,'g.');  

%% Q3b) - 2D plot of 2 principal components
Ureduce2 = U(:,1:2);
Sreduce2 = S(1:2,1:2);
z2 = Ureduce2*Sreduce2;
y = meas(:,5);
datax1 = z2(1:50,1);
datax2 = z2(51:100,1);
datax3 = z2(101:150,1);
datay1 = z2(1:50,2);
datay2 = z2(51:100,2);
datay3 = z2(101:150,2);

figure(8);
hold on
grid on
plot(datax1,datay1,'r.');  
plot(datax2,datay2,'b.'); 
plot(datax3,datay3,'g.');  
hold off

%% Q3b) - 3D plot of 3 principal components
Ureduce3 = U(:,1:3);
Sreduce3 = S(1:3,1:3);
z3 = Ureduce3*Sreduce3;

datax1 = z3(1:50,1);
datax2 = z3(51:100,1);
datax3 = z3(101:150,1);
datay1 = z3(1:50,2);
datay2 = z3(51:100,2);
datay3 = z3(101:150,2);
dataz1 = z3(1:50,3);
dataz2 = z3(51:100,3);
dataz3 = z3(101:150,3);

figure(9);
grid on
scatter3(datax1,datay1,dataz1,'r.');
hold on
scatter3(datax2,datay2,dataz2,'b.'); 
scatter3(datax3,datay3,dataz3,'g.');  
hold off

% The relative positions of the green, blue and red points are about the
% same in all three graphs. This pattern remains regular even when going
% from 1 to 3 principal components.

%% Q4a) - k-means on Principal Components

clear all
clc

rng(2021);
load iris;
X = meas(:,1:4);
[U,S,V]=svd(X);
z = U*S;

clust = zeros(size(z,1),6);
for i=1:6
clust(:,i) = kmeans(z,i,'emptyaction','singleton',...
        'replicate',5);
end
va = evalclusters(z,clust,'CalinskiHarabasz')

% Optimal k is 3

k = 3;
[id,C] = kmeans(z,k,'emptyaction','singleton',...
    'replicate',5);

% Reordering clusters to match previous flower numbering
idx = zeros(150:1);
for i = 1:150
    if id(i)==2
        idx(i)=1;
    elseif id(i)==1
        idx(i)=3;
    else
        idx(i)=2;
    end
end

figure(10);
plot3(z(idx==1,1),z(idx==1,2),z(idx==1,3),'r.','MarkerSize',12)
hold on
grid on
plot3(z(idx==2,1),z(idx==2,2),z(idx==2,3),'b.','MarkerSize',12)
plot3(z(idx==3,1),z(idx==3,2),z(idx==3,3),'g.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','SE')
title 'Cluster Assignments and Centroids'
hold off

%%
y = meas(:,5);

figure(11);
C = confusionchart(y,idx);
disp('Confusion matrix is:'); disp(C)
C.RowSummary = 'row-normalized';
C.ColumnSummary = 'column-normalized';

%%
p1 = 1;
r1 = 1;
p2 = 0.96;
r2 = 0.774;
p3 = 0.72;
r3 = 0.947;
f1 = 2*(p1*r1)/(p1+r1);
f2 = 2*(p2*r2)/(p2+r2);
f3 = 2*(p3*r3)/(p3+r3);
disp('Precision and Recall for class 1 are:'); disp(p1); disp(r1);
disp('Precision and Recall for class 2 are:'); disp(p2); disp(r2);
disp('Precision and Recall for class 3 are:'); disp(p3); disp(r3);
disp('F measure for class 1 is:'); disp(f1);
disp('F measure for class 2 is:'); disp(f2);
disp('F measure for class 3 is:'); disp(f3);

Acc = (50+48+36)/150;
disp('Accuracy of the system is:');disp(Acc);

%% Q4a) - Normalization

zn = normalize(z);
[id,C] = kmeans(zn,k,'emptyaction','singleton',...
    'replicate',5);
% Reordering clusters to match previous flower numbering
rng(2021);
idx = zeros(150:1);
for i = 1:150
    if id(i)==1
        idx(i)=1;
    elseif id(i)==2
        idx(i)=2;
    else
        idx(i)=3;
    end
end

figure(12);
plot3(zn(idx==1,1),zn(idx==1,2),zn(idx==1,3),'r.','MarkerSize',12)
hold on
plot3(zn(idx==2,1),zn(idx==2,2),zn(idx==2,3),'b.','MarkerSize',12)
plot3(zn(idx==3,1),zn(idx==3,2),zn(idx==3,3),'g.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','SE')
title 'Cluster Assignments and Centroids'
hold off

%%
y = meas(:,5);

figure(13);
C = confusionchart(y,idx);
disp('Confusion matrix is:'); disp(C)
C.RowSummary = 'row-normalized';
C.ColumnSummary = 'column-normalized';

% It looks like normalization has caused the accuracy of the method to
% deprove, based on the confusion matrix and precision/recall details.

%% Q4b) - k-means on 1 principal component

clear all
clc

rng(2021);
load iris;
X = meas(:,1:4);
[U,S,V]=svd(X);

% Assuming k=3 optimal clusters for all principal components

k = 3;

Ureduce1 = U(:,1:1);
Sreduce1 = S(1,1);
z1 = Ureduce1.*Sreduce1;
y = meas(:,5);

[id,C] = kmeans(z1,k,'emptyaction','singleton',...
    'replicate',5);
% Reordering clusters to match previous flower numbering
rng(2021);
idx = zeros(150:1);
for i = 1:150
    if id(i)==3
        idx(i)=1;
    elseif id(i)==2
        idx(i)=3;
    else
        idx(i)=2;
    end
end

figure(14);
plot(z1(idx==1,1),'r.','MarkerSize',12)
hold on
plot(z1(idx==2,1),'b.','MarkerSize',12)
plot(z1(idx==3,1),'g.','MarkerSize',12)
plot(C,'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','SE')
title 'Cluster Assignments and Centroids'
hold off

%%
figure(15);
C = confusionchart(y,idx);
disp('Confusion matrix is:'); disp(C)
C.RowSummary = 'row-normalized';
C.ColumnSummary = 'column-normalized';

%%
p1 = 1;
r1 = 0.926;
p2 = 0.82;
r2 = 0.732;
p3 = 0.70;
r3 = 0.875;
f1 = 2*(p1*r1)/(p1+r1);
f2 = 2*(p2*r2)/(p2+r2);
f3 = 2*(p3*r3)/(p3+r3);
disp('Precision and Recall for class 1 are:'); disp(p1); disp(r1);
disp('Precision and Recall for class 2 are:'); disp(p2); disp(r2);
disp('Precision and Recall for class 3 are:'); disp(p3); disp(r3);
disp('F measure for class 1 is:'); disp(f1);
disp('F measure for class 2 is:'); disp(f2);
disp('F measure for class 3 is:'); disp(f3);

Acc = (50+41+35)/150;
disp('Accuracy of the system is:');disp(Acc);

%% Q4b) - k-means on 2 principal components

clear all
clc

rng(2021);
load iris;
X = meas(:,1:4);
[U,S,V]=svd(X);

% Assuming k=3 optimal clusters for all principal components

k = 3;

Ureduce2 = U(:,1:2);
Sreduce2 = S(1:2,1:2);
z2 = Ureduce2*Sreduce2;
y = meas(:,5);

[id,C] = kmeans(z2,k,'emptyaction','singleton',...
    'replicate',5);

% Reordering clusters to match previous flower numbering
idx = zeros(150:1);
for i = 1:150
    if id(i)==3
        idx(i)=1;
    elseif id(i)==1
        idx(i)=2;
    else
        idx(i)=3;
    end
end

figure(16);
plot(z2(idx==1,1),z2(idx==1,2),'r.','MarkerSize',12)
hold on
grid on
plot(z2(idx==2,1),z2(idx==2,2),'b.','MarkerSize',12)
plot(z2(idx==3,1),z2(idx==3,2),'g.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','SE')
title 'Cluster Assignments and Centroids'
hold off

%%
figure(17);
C = confusionchart(y,idx);
disp('Confusion matrix is:'); disp(C)
C.RowSummary = 'row-normalized';
C.ColumnSummary = 'column-normalized';

%%
p1 = 1;
r1 = 1;
p2 = 0.94;
r2 = 0.77;
p3 = 0.72;
r3 = 0.923;
f1 = 2*(p1*r1)/(p1+r1);
f2 = 2*(p2*r2)/(p2+r2);
f3 = 2*(p3*r3)/(p3+r3);
disp('Precision and Recall for class 1 are:'); disp(p1); disp(r1);
disp('Precision and Recall for class 2 are:'); disp(p2); disp(r2);
disp('Precision and Recall for class 3 are:'); disp(p3); disp(r3);
disp('F measure for class 1 is:'); disp(f1);
disp('F measure for class 2 is:'); disp(f2);
disp('F measure for class 3 is:'); disp(f3);

Acc = (50+47+36)/150;
disp('Accuracy of the system is:');disp(Acc);

%% Q4b) - k-means on 3 principal components

clear all
clc

rng(2021);
load iris;
X = meas(:,1:4);
[U,S,V]=svd(X);
Ureduce3 = U(:,1:3);
Sreduce3 = S(1:3,1:3);
z3 = Ureduce3*Sreduce3;

k = 3;
[id,C] = kmeans(z3,k,'emptyaction','singleton',...
    'replicate',5);

% Reordering clusters to match previous flower numbering
idx = zeros(150:1);
for i = 1:150
    if id(i)==1
        idx(i)=1;
    elseif id(i)==2
        idx(i)=3;
    else
        idx(i)=2;
    end
end

figure(18);
plot3(z3(idx==1,1),z3(idx==1,2),z3(idx==1,3),'r.','MarkerSize',12)
hold on
grid on
plot3(z3(idx==2,1),z3(idx==2,2),z3(idx==2,3),'b.','MarkerSize',12)
plot3(z3(idx==3,1),z3(idx==3,2),z3(idx==3,3),'g.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','SE')
title 'Cluster Assignments and Centroids'
hold off

%%
y = meas(:,5);

figure(19);
C = confusionchart(y,idx);
disp('Confusion matrix is:'); disp(C)
C.RowSummary = 'row-normalized';
C.ColumnSummary = 'column-normalized';

%%
p1 = 1;
r1 = 1;
p2 = 0.96;
r2 = 0.774;
p3 = 0.72;
r3 = 0.947;
f1 = 2*(p1*r1)/(p1+r1);
f2 = 2*(p2*r2)/(p2+r2);
f3 = 2*(p3*r3)/(p3+r3);
disp('Precision and Recall for class 1 are:'); disp(p1); disp(r1);
disp('Precision and Recall for class 2 are:'); disp(p2); disp(r2);
disp('Precision and Recall for class 3 are:'); disp(p3); disp(r3);
disp('F measure for class 1 is:'); disp(f1);
disp('F measure for class 2 is:'); disp(f2);
disp('F measure for class 3 is:'); disp(f3);

Acc = (50+48+36)/150;
disp('Accuracy of the system is:');disp(Acc);

% From this, we can see that k-means accuracy increases as number of
% principal components increases, but to a decreasing extent. The gain from
% 1 component to 2 components is much greater than the gain from 2 to 3
% components.