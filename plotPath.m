clear
close

% 将数据加载到 MATLAB 中 
loadGroundTruthAGL

% 绘制真实位置（Ground Truth）
plot3(x_gt, y_gt, z_gt, '.')
grid on
hold on

% 绘制 GPS 位置
plot3(x_gps, y_gps, z_gps, 'or')
axis equal
axis vis3d