# AGZ 数据集说明 (AGZ Dataset Readme)

---

### ⚠️ 重要提示：版权信息
- **MAV 图像数据** (位于 `./AGZ/MAV Images/` 和 `./AGZ/MAV Images Calib/`)：版权归作者 Andras L. Majdik, Yves Albers-Schoenberg 和 Davide Scaramuzza 所有。这些数据可不受限制地用于学术研究。
- **街景图像** (位于 `./AGZ/Street View Images/`)：版权归 Google Inc. 所有。更多详情请参考 [www.google.com/streetview](http://www.google.com/streetview)。

---

## 📂 日志文件 (Log Files)

这些文件存储在 `./AGZ/Log Files/` 目录下。

### 1. `BarometricPressure.csv` (气压计数据)
包含机载气压传感器的数据：
- **列信息**：1: 时间戳 (Timestamp), 2: 气压 (Pressure), 3: 海拔 (Altitude), 4: 温度 (Temperature)

### 2. `GroundTruthAGL.csv` (地面真实位置 AGL)
包含相机位置的真实值：
- **列信息**：
  - 1: `imgid` (MAV 图像 ID)
  - 2-4: `x_gt`, `y_gt`, `z_gt` (相机位置真实值 X, Y, Z)
  - 5: `omega_gt` (偏航角 Yaw, 单位: 度)
  - 6: `phi_gt` (俯仰角 Pitch, 单位: 度)
  - 7: `kappa_gt` (翻滚角 Roll, 单位: 度)
  - 8-10: `x_gps`, `y_gps`, `z_gps` (GPS 相机位置 X, Y, Z)
- **坐标系**：所有位置值均采用 WGS 84 / UTM zone 32N 坐标系。可以使用 `plotPath.m` 在 Matlab 中进行可视化。

### 3. `GroundTruthAGM` (最近邻街景图像)
- **列信息**：1: 时间戳, 2: `imgid` (MAV 图像 ID), 3-5: `svid_1, svid_2, svid_3` (最近、次近和第三近的 Google 街景图像 ID)
- **说明**：距离是根据原始 GPS 标签计算得出的。

### 4. `OnbordGPS.csv` (机载 GPS 数据)
包含 MAV 机载 GPS 接收器的数据：
- **列信息**：
  - 1: 时间戳, 2: `imgid` (MAV 图像 ID)
  - 3: `lat` (纬度，单位：1E7 度)
  - 4: `lon` (经度，单位：1E7 度)
  - 5: `alt` (海拔，高于平均海平面 MSL，单位：1E3 米)
  - 6: `s_variance_m_s` (速度精度估计，单位：m/s)
  - 7: `c_variance_rad` (航向精度估计，单位：rad)
  - 8: `fix_type` (定位类型：0-1: 未定位, 2: 2D 定位, 3: 3D 定位)
  - 9: `eph_m` (水平定位精度 HDOP，单位：m)
  - 10: `epv_m` (垂直定位精度 VDOP，单位：m)
  - 11-13: `vel_n_m_s`, `vel_e_m_s`, `vel_d_m_s` (北、东、地向地面速度，单位：m/s)
  - 14: `num_sat` (可见卫星数量)
- [更多详情](https://home.hibu.no/AtekStudenter1212/doxygen/bb_handler/structvehicle__gps__position__s.html)

### 5. `OnboardPose.csv` (机载姿态数据)
包含原始传感器数据及由 PIXHAWK 自动驾驶仪估算的姿态：
- **列信息**：1: 时间戳, 2-4: 角速度 (Omega_x/y/z), 5-7: 加速度 (Accel_x/y/z), 8-10: 速度 (Vel_x/y/z), 11-13: 加速度偏差 (AccBias_x/y/z), 14: 方位角 (Azimuth), 15-18: 四元数姿态 (Attitude_w/x/y/z), 19: 高度 (Height), 20: 海拔 (Altitude), 21: 俯仰角 (veh_pitch), 22-24: 系留相关参数, 25: GPS 是否开启

### 6. `RawAccel.csv` & `RawGyro.csv` (原始加速度/陀螺仪数据)
- **列信息**：1: 时间戳, 2: 错误计数, 3-5: x, y, z, 6: 温度, 7: 范围 (rad/s), 8: 比例, 9-11: 原始 x, y, z, 12: 原始温度

### 7. `StreetViewGPS.csv` (街景 GPS 数据)
包含每张数据库街景图像的 GPS 数据（地理标签）：
- **列信息**：1: 纬度, 2: 经度, 3: 偏航角, 4: 倾斜偏航角, 5: 倾斜俯仰角, 6: 辅助变量

---

## 🖼️ 图像文件夹

- **`./AGZ/MAV Images/`**：包含在瑞士苏黎世由微型飞行器 (MAV) 拍摄的 81,169 张图像。
- **`./AGZ/MAV Images Calib/`**：包含 30 张 MAV 拍摄的图像，用于计算相机内参。
- **`./AGZ/Street View Images/`**：包含与 MAV 拍摄区域对应的 113 张街景图像。

---

## 🛠️ 其他文件与脚本

- **`./AGZ/calibration_data.npz`**：使用校准图像计算得出的相机内参。
- **`./AGZ/loadGroundTruthAGL.m`**：用于 `plotPath.m` 加载数据到 Matlab 的脚本。
- **`./AGZ/plotPath.m`**：在 Matlab 中可视化轨迹的脚本。
- **`./AGZ/write_ros_bag.py`**：将数据写入 ROS bag 文件的脚本。执行方式：`python write_ros_bag.py`
