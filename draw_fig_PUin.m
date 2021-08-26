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