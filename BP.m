clc
clear
close all 


while 1==1
%%

load data_xy.mat
data_y = data(1:50000,1:1);
data_x = data(1:50000,2:2);
%构建
train_x = [data_x]';%数据
train_y = [data_y]';%标签

%构建网络
net = newff(train_x,train_y, [10], { 'logsig' 'purelin' } , 'traingdx' , 'learngdm') ;%输入数据为 特征数*数据个数，输出为 类别向量*数据个数
net.trainParam.showWindow = 0;% 不显示窗口
net.trainparam.epochs = 5000;%允许最大训练步数500步
net.trainparam.goal = 0.00000001 ;%训练目标最小误差0.01
net.trainParam.lr = 0.0000001 ;%学习速率0.05
%% 开始训练
net = train( net, train_x,train_y);
close all 
%% 仿真测试
predict_y = sim( net,train_x) ;

SST = sum((train_y-mean(train_y)).^2);
SSR = sum((predict_y'-mean(train_y)).^2);

R2 = SSR./SST

if 0.998 <R2&&R2< 1.002
    save GoodModel.mat
    x = 1:50000;
    plot(x,train_y,'r',x,predict_y,'g');
    break
end
end
