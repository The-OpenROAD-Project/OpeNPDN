close all ;clear ;
% figure;
% J  = csvread("work/current_map_processed.csv");
% surf(J,'linestyle','none');
% cb1 = colorbar;
% colormap(jet(256))
% xlabel("Width")
% ylabel("Height")
% cb1.Label.String = "Current(A)"

figure;
V  = csvread("output/IR_drop.csv");
V(:,1) = V(:,1)*1e6;
V(:,2) = V(:,2)*1e6;

xmin = min(V(:,1));
xmax = max(V(:,1));
ymax = max(V(:,2));
ymin = min(V(:,2));

[xq,yq] = meshgrid(xmin:1:xmax,ymin:1:ymax);
vq = griddata(V(:,1),V(:,2),V(:,3),xq,yq);
surf(vq,'linestyle','none');
colormap(jet(256))
xlabel("Width")
ylabel("Height")
cb = colorbar;
cb.Label.String = "IR drop(V)"

disp("max drop is: ")
max(V(:,3))