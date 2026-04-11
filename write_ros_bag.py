import rosbag
import rospy
import roslib
import tf
from std_msgs.msg import UInt32, String, Int32, Int8MultiArray
from sensor_msgs.msg import NavSatFix, Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import csv
import itertools
import ImageFile
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('test')

step = 15
stop_at = 350
bag = rosbag.Bag('AGZ.bag', 'w')

# 读取相机校正数据
Caminfo = CameraInfo()
data = np.load('calibration_data.npz')
# print(data['intrinsic_matrix']) # 打印相机内参矩阵

# 读取日志文件
f = open("Log Files/OnboardGPS.csv")
csv_gps = csv.reader(f)
header = csv_gps.next()
f = open("Log Files/GroundTruthAGM.csv")
csv_gt = csv.reader(f)
header = csv_gt.next()
f = open("Log Files/GroundTruthAGL.csv")
csv_gtl = csv.reader(f)
header = csv_gtl.next()

# 初始化变量
gps_lat_prev = 0
gps_lon_prev = 0
gps_seq = 0
best_prev = 0
stv_seq = 0
cal = 0

for row_gps, row_gt in itertools.izip(csv_gps, csv_gt):
    imgid = int(row_gps[1])
    print(row_gps[1])
#    print(row_gt[1])	
    t = rospy.Time()
    i = int(row_gps[0]) # i 以微秒为单位
    t = rospy.Duration(0,i)*1000

    ## 写入机载 GPS 数据
    if (gps_lat_prev != float(row_gps[2]) or gps_lon_prev != float(row_gps[3])):    
        gps_seq = gps_seq + 1
        Gps = NavSatFix()
        Gps.header.seq = int(gps_seq)
        Gps.header.stamp = t
        Gps.header.frame_id = 'gps'
        Gps.status.service = 1
        Gps.latitude = float(row_gps[2])
        Gps.longitude = float(row_gps[3])
        Gps.altitude = float(row_gps[4])
        bag.write('gps', Gps, t)
        # print(Gps)
    gps_lat_prev = float(row_gps[2])
    gps_lon_prev = float(row_gps[3])

    ## 写入地面真实（Ground Truth）街景图片 ID（前 3 位）
    Igt = Int8MultiArray()
    Igt.data = [int(row_gt[2]), int(row_gt[3]), int(row_gt[4])]
    bag.write('groundtruth/sv_id', Igt, t) 

    ## 写入最佳街景图片
    best = int(row_gt[2])
    if best_prev != best:
        stv_seq = stv_seq + 1 
        img_cv_sv = cv2.imread( "Street View Images/left-" + '{0:03d}'.format(best) + ".jpg", 1) 
        br = CvBridge()
        Img = Image();
        Img = br.cv2_to_imgmsg(img_cv_sv, "bgr8")
        Img.header.seq = stv_seq
        Img.header.stamp = t
        Img.header.frame_id = 'streetview'
        bag.write('groundtruth/image', Img, t)
    best_prev = best

    ## 写入航拍图片
    img_cv = cv2.imread( "MAV Images/" + '{0:05d}'.format(int(row_gps[1])) + ".jpg" , 1)
    br = CvBridge()
    Img = Image();
    Img = br.cv2_to_imgmsg(img_cv, "bgr8")
    Img.header.seq = int(row_gps[1])
    # print(Img.header.seq)
    Img.header.stamp = t
    # print(t)
    Img.header.frame_id = 'camera'
    bag.write('camera/image', Img, t)

    ## 写入航拍图片的地面真实姿态
    if (imgid - 1) % 30 == 0:
        row_gtl = csv_gtl.next()
        # print(row_gtl[0])
        Pgtl = PoseStamped()
        Pgtl.pose.position.x = float(row_gtl[1])
        Pgtl.pose.position.y = float(row_gtl[2])
        Pgtl.pose.position.z = float(row_gtl[3])
        quaternion = tf.transformations.quaternion_from_euler(float(row_gtl[4]), float(row_gtl[5]), float(row_gtl[6]))
        Pgtl.pose.orientation.x = quaternion[0]
        Pgtl.pose.orientation.y = quaternion[1]
        Pgtl.pose.orientation.z = quaternion[2]
        Pgtl.pose.orientation.w = quaternion[3]
        Pgtl.header.seq = int(row_gtl[0])
        Pgtl.header.stamp = t
        Pgtl.header.frame_id = 'world'
        bag.write('groundtruth/pose', Pgtl, t)


    ## 写入相机校正数据
    if cal < 1:
        Caminfo.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        Caminfo.P = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        Caminfo.D = np.asarray(data['distCoeff']).reshape(-1)
        Caminfo.K = np.asarray(data['intrinsic_matrix']).reshape(-1)
        Caminfo.binning_x = 1
        Caminfo.binning_y = 1
        img_cv_h, img_cv_w = img_cv.shape[:2]
        Caminfo.width = img_cv_w     
        Caminfo.height = img_cv_h
        Caminfo.distortion_model = 'plumb_bob'
        bag.write('camera/camera_info', Caminfo, t)
        # cal = 1; # 如果只需执行一次，请取消注释放行

    if imgid > stop_at - step:
        for num in range(1,81169 - stop_at):
            try:
                csv_gt.next()
                csv_gps.next()
            except StopIteration:
                print('文件结束')

    for num in range(1,step):
        try:
            csv_gt.next()
            csv_gps.next()
        except StopIteration:
            print('文件结束')

bag.close()
