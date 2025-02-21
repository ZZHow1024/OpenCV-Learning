# 01OpenCV简介

参考课程：

**【*黑马程序员* OpenCV入门教程】**

[https://www.bilibili.com/video/BV1Fo4y1d7JL]

@ZZHow(ZZHow1024)

# 1.1图像处理简介

- 图像是指能在人的视觉系统中产生视觉印象的客观对象，包括自然景物、拍摄到的图片、用数学方法描述的图形
- 模拟图像 — 发展 —> 数字图像
    - 模拟图像：连续存储的数据
    - 数字图像：分级存储的数据
- 数字图像的表示
    - 位数：计算机采用 0 / 1 编码的系统，数字图像也是利用 0 / 1 来记录信息。日常接触的图像都是 8 位数图像，包含 0～255 灰度，其中 0 表示最黑，1 表示最白
    - 人眼对灰度的敏感值，在 16 位到 32 位之间
- 图像的分类
    - 二值图像：一幅二值图像的二维矩阵仅由 0、1 两个值构成，0 代表黑色，1 代白色
        - 由于每一像素（矩阵中每一元素）取值仅有 0、1 两种可能，所以计算机中二值图像的数据类型通常为 1 个二进制位。二值图像通常用于文字、线条图的扫描识别(OCR)和掩膜图像的存储
    - 灰度图：每个像素只有一个采样颜色的图像
        - 这类图像通常显示为从最暗黑色到最亮的白色的灰度
        - 灰度图像在黑色与白色之间还有许多级的颜色深度
        - 用于显示的灰度图像通常用每个采样像素 8 位的非线性尺度来保存，可以有 256 级灰度（如果用16位，则有 65536 级）
    - 彩色图：每个像素通常是由红(R)、绿(G)、蓝(B)三个分量来表示的，分量介于(0, 255)
        - 它分别用红(R)、绿(G)、蓝(B)三原色的组合来表示每个像素的颜色
        - RGB 图像的数据类型一般为 8 位无符号整型，通常用于表示和存放真彩色图像

# 1.2OpenCV简介

- OpenCV 是一款由 Intel 公司俄罗斯团队发起并参与和维护的一个计算机视觉处理开源软件库，支持与计算机视觉和机器学习相关的众多算法，并且正在日益扩展
- OpenCV 的优势：
    - 支持的编程语言丰富
    - 支持跨平台
    - 丰富的 API
- OpenCV-Python：OpenCV-Python 是一个 Python 绑定库，旨在解决计算机视觉问题
    - OpenCV-Python 使用 NumPy，NumPy 是一个局度仇化的数据库操作库，具有 MATLAB 风格的语法
    - 所有 OpenCV 数组结构都转换为 NumPy 数组，与使用 NumPy 的其他库集成更容易
- 安装
    - `pip install opencv-python`
    - `pip install opencv-contrib-python`
- 读图片测试
    
    ```python
    import cv2
    
    image = cv2.imread("OpenCV.png")
    cv2.imshow("image", image)
    cv2.waitKey(0)
    ```
    
    - 案例演示：[**test_opencv.py**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/01OpenCV%E7%AE%80%E4%BB%8B/test_opencv.py)

# 1.3OpenCV的模块

- 最基础的模块
    - **core 模块**实现了最核心的数据结构及其基本运算，如绘图函数、数组操作相关函数等
    - **highgui 模块**实现了视频与图像的读取、显示、存储等接口
    - **imgproc 模块**实现了图像处理的基础方法，包括图像滤波、图像的几何变换、平滑、阈值分割、形态学处理、边缘检测、目标检测、运动分析和对象跟踪等
- 更高层次应用的模块
    - **features2d 模块**用于提取图像特征以及特征匹配，**nonfree 模块**实现了一些专利算法
    - **objdetect 模块**实现了一些目标检测的功能，经典的基于 Haar、LBP 特征的人脸检测，基于 HOG 的行人、汽车等目标检测，分类器使用 Cascade Classification（级联分类）和 Latent SVM 等
    - **stitching 模块**实现了图像拼接功能
    - **FLANN 模块(Fast Library for Approximate Nearest Neighbors)**，包含快速近似最近邻搜索 FLANN 和聚类 Clustering 算法
    - **ml 模块**机器学习模块（SVM，决策树，Boosting 等等）
    - **photo 模块**包含图像修复和图像去噪两部分
    - **video 模块**针对视频处理，如背景分离，前景检测、对象跟踪等
    - **calib3d 模块**即 Calibration（校准）3D，这个模块主要是相机校准和三维重建相关的内容
    - **G-API 模块**包含超高效的图像处理 pipeline 引擎
