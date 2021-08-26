%clear all
close all
load('topo_PU_inside.mat')
radius=50;
num_PU=5;
PU_pos=[40 0;-30 15;-20 20;15 -15;0 -30];
for i=1:20
    for i_PU = 1:5
        dis_PU(i_PU,i)=norm(SU_pos(i,:)-PU_pos(i_PU));
    end
end
save('topo_PU_inside.mat','SU_pos','PU_pos','weight_full','dis_PU_full','dis_SU_full')
figure(1);
N=20;
for i=1:360
    cir_x(i)=radius*cos(i/360*2*pi);
    cir_y(i)=radius*sin(i/360*2*pi);
end
h1=plot(0,0,'ro','MarkerSize',16);hold on
h2=plot(SU_pos(1:N,1),SU_pos(1:N,2),'k.','MarkerSize',20);hold on;
for i_PU=1:num_PU
    h3=plot(PU_pos(i_PU,1),PU_pos(i_PU,2),'g*','MarkerSize',12);hold on
end
plot(cir_x,cir_y,'--');hold on
le1=legend([h1 h2 h3],'Pico BS','SUs','PUs','Location','NorthEast');
set(gca,'fontsize',26);
xlim([-62 62]);ylim([-62 62]);
ax = gca;
ax.XTick=-60:30:60;
ax.YTick=-60:30:60;

vio_PUin=[
    0    0.0005    0.0065    0.0358    0.0741    0.1141    0.1541;
    0    0.0001    0.0002    0.0009    0.0021    0.0035    0.0050;
    0    0.0015    0.0105    0.0417    0.0771    0.1125    0.1481
    0    0.0002    0.0012    0.0047    0.0101    0.0160    0.0236
];

obj_PUin=[
        0.4968    0.7348    0.8355    0.9305    0.9850    1.0236    1.0551
    0.4469    0.6551    0.7420    0.8238    0.8698    0.9026    0.9286
    0.3791    0.5981    0.6996    0.8037    0.8667    0.9125    0.9506
    0.3333    0.5291    0.6209    0.7141    0.7696    0.8105    0.8441
];
epi=[0.01 0.05 0.1:0.1:0.5];
figure(2);
line_type={'b-s','b:p','g-o','g:v'};
for i=1:4
    h(i)=plot(epi,vio_PUin(i,:),line_type{i},'Linewidth',2,'markersize',10);hold on
end
plot(epi,epi,'r-','Linewidth',2);
set(gca,'fontsize',20,'LineWidth',2); 
le1=legend(h,'PU (-20, 20), Ind.','PU (15, -15), Ind.','PU (-20, 20), Corr.','PU (15, -15), Corr.');
set(le1,'Location','NorthEast');
xlabel('Risk level \epsilon');ylabel('Threshold violation probability')
xlim([0.01 0.5])
ylim([0 0.25])
ax = gca;
ax.XTick= [0.01 0.1 0.2:0.1:0.5];
ax.YTick=[0:0.05:0.25];
% Create textarrow
annotation('textarrow',[0.35 0.394642857142857],...
    [0.733333333333333 0.7],'String',{'Risk level \epsilon'},'LineWidth',2,...
    'HeadLength',12,...
    'FontSize',20);

figure(3);
h3=plot(epi,obj_PUin);
NameArray = {'LineStyle','Color','Marker'};
ValueArray = {
    '-','k','x';
    '-.','b','s';
    '--','k','+';
    ':','b','p';
    };
set(h3,NameArray,ValueArray);
set(h3,'linewidth',2,'markersize',10);
set(gca,'fontsize',20);
xlabel('Risk level \epsilon');ylabel('Objective (bps/Hz)')
xlim([0.01 0.5])
ylim([0 1.1])
ax = gca;
ax.XTick= [0.01 0.1 0.2:0.1:0.5];
ax.YTick=[0:0.2:1.1];
legend('GUC, Ind.','Optimal, Ind.','GUC, Corr.','Optimal, Corr.','Location','SouthEast')

saveas(figure(1),'topo_PUin.eps','epsc')
saveas(figure(2),'vio_PUin.eps','epsc')
 saveas(figure(3),'obj_PUin.eps','epsc')
