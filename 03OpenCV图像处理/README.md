# 03OpenCV图像处理

参考课程：

**【*黑马程序员* OpenCV入门教程】**

[https://www.bilibili.com/video/BV1Fo4y1d7JL]

@ZZHow(ZZHow1024)

# 1.1几何变换

- 图像缩放
    - 对图像的大小进行调整，即使图像放大或缩小
    
    ```python
    cv2.resize(src, dsize, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    ```
    
    - 参数
        - src：输入图像
        - dsize：绝对尺寸，直接指定调整后图像的大小
        - fx, fy：相对尺寸，将 dsize 设置为 None，然后将 fx 和 fy 设置为比例因子即可
        - interpolation：插值方法
            
            
            | 插值 | 含义 |
            | --- | --- |
            | cv2.INTER_LINEAR | 双线性插值法 |
            | cv2.INTER_NEAREST | 量近邻插值 |
            | cv2.INTER_AREA | 像素区域重采样**（默认）** |
            | cv2.INTER_CUBIC | 双三次插值 |
- 图像平移
    - 将图像按照指定方向和距离，移动到相应的位置
    
    ```python
    cv2.warpAffine(image, M, dsize)
    ```
    
    - 参数
        - image：输入图像
        - M：2 × 3 移动矩阵
            - 对于 $(x, y)$ 处的像素点，要把它移动到 $(x + t_x, y + t_y)$ 处时，$M$ 矩阵应如下设置
                
                $M = \begin{bmatrix}
                   1 & 0 & t_x \\
                   0 & 1 & t_y
                \end{bmatrix}$
                
            - **注意：将 $M$ 设置为 `np.float32` 类型的 NumPy 数组**
        - dsize：输出图像的大小
    - **注意：输出图像的大小应该是 (宽度, 高度) 的形式，width=列数，height=行数**
- 图像旋转
    - 将图像按照某个位置转动一定角度的过程，旋转中图像仍保持这原始尺寸
    - 假设图像逆时针旋转 $\theta$，则根据坐标转换可得旋转转换为
        
        $\begin{cases}   x' = r cos(\alpha - \theta) \\   y' = r sin(\alpha - \theta)\end{cases}$
        
        其中 $r = \sqrt{x^2 + y^2}, sin\alpha = \frac{y}{\sqrt{x^2 + y^2}}, cos\alpha = \frac{x}{\sqrt{x^2 + y^2}}$
        
        代入公式有
        
        $\begin{cases}   x' = x cos\theta + y sin\theta \\   y' = -x sin\theta + ycos\theta\end{cases}$
        
        也可以写成
        
        $\begin{bmatrix}   x' & y' & 1\end{bmatrix} = \begin{bmatrix}   x & y & 1\end{bmatrix}\begin{bmatrix}   cos\theta & -sin\theta & 0 \\   sin\theta & cos\theta & 0 \\   0 & 0 & 1\end{bmatrix}$
        
        原点修正
        
        $\begin{bmatrix}   x'' & y'' & 1\end{bmatrix} = \begin{bmatrix}   x' & y' & 1\end{bmatrix}\begin{bmatrix}   1 & 0 & 0 \\   0 & -1 & 0 \\   left & top & 1\end{bmatrix} = \begin{bmatrix}   x & y & 1\end{bmatrix}\begin{bmatrix}   cos\theta & -sin\theta & 0 \\   sin\theta & cos\theta & 0 \\   0 & 0 & 1\end{bmatrix}\begin{bmatrix}   1 & 0 & 0 \\   0 & -1 & 0 \\   left & top & 1\end{bmatrix}$
        
    
    ```python
    cv2.getRotationMatrix2D(center, angle, scale)
    ```
    
    - 参数
        - center：旋转中心
        - angle：旋转角度
        - scale：缩放比例
    - 返回
    M：旋转矩阵，调用 `cv.warpAffine()` 完成图像的旋转
- 仿射变换
    - 图像的仿射变换涉及到图像的形状位置角度的变化，是深度学习预处理中常到的功能，仿射变换主要是对图像的缩放，旋转，翻转和平移等操作的组合
    - 在OpenCV中，仿射变换的矩阵是一个 2 × 3 的矩阵
        
        $M = \begin{bmatrix}   A & B\end{bmatrix} = \begin{bmatrix}   a_{00} & a_{01} & b_0 \\   a_{10} & a_{11} & b_1 \\\end{bmatrix}$
        
    - 其中左边的 2 × 2 子矩阵 $A$ 是线性变换矩阵，右边的 2 × 1 子矩阵 $B$ 是平移项
        
        $A = \begin{bmatrix}   a_{00} & a_{01} \\   a_{10} & a_{11} \\\end{bmatrix},B = \begin{bmatrix}   b_0 \\   b_1 \\\end{bmatrix}$
        
    - 对于图像上的任一位置 $(x, y)$，仿射变换执行的是如下的操作
        
        $T_{affine} =A\begin{bmatrix}   x \\   y \\\end{bmatrix} + B = M\begin{bmatrix}   x \\   y \\   1 \\\end{bmatrix}$
        
    - **注意：对于图像而言，宽度方向是 x，高度方向是 y，坐标的顺序和图像像素对应下标一致，原点的位置不是左下角而是右上角，y 的方向也不是向上，而是向下**
    - 在 OpenCV 中 `cv2.getAffineTransform` 会创建一个 2 × 3 的矩阵，最后这个矩阵会被传给函数 `cv2.warpAffine`
- 透射变换
    - 透射变换是视角变化的结果，是指利用透视中心、像点、目标点三点共线的条件，按透视旋转定律使承影面（透视面）绕迹线（透视轴）旋转某一角度，破坏原有的投影光线束，仍能保持承影面上投影几何图形不变的变换
    - 它的本质将图像投影到一个新的视平面，其通用变换公式为
        
        $\begin{bmatrix}   x' & y' & z'\end{bmatrix} = \begin{bmatrix}   u & v & w\end{bmatrix}\begin{bmatrix}   a_{00} & a_{01} & a_{02} \\   a_{10} & a_{11} & a_{12} \\   a_{20} & a_{21} & a_{22} \\\end{bmatrix}$
        
        其中，$(u, v)$ 是原始的图像像素坐标，$w$ 取值为 1，$(x = x' / z', y = y’ / z’)$ 是透射变换后的结果
        
        后面的矩阵称为透视变换矩阵，一般情况下，我们将其分为三部分
        
        $T = \begin{bmatrix}   a_{00} & a_{01} & a_{02} \\   a_{10} & a_{11} & a_{12} \\   a_{20} & a_{21} & a_{22} \\\end{bmatrix} = \begin{bmatrix}   T_1 & T_2 \\   T_3 & a_{22} \\\end{bmatrix}$
        
        其中，$T_1$ 表示对图像进行线性变换，$T_2$ 对图像进行平移，$T_3$ 表示对图像进行投射变换，$a_{22}$ 一般设为 1
        
    - 在 OpenCV 中，我们要找到四个点，其中任意三个不共线，然后获取变换矩阵 $T$，再进行透射变换
        - 通过函数 `cv2.getPerspectiveTransform` 找到变换矩阵，将 `cv2.warpPerspective` 应用于此 3 × 3 变换矩阵
- 图像金字塔
    - 图像多尺度表达的一种，最主要用于图像的分割，是一种以多分辨率来解释图像的有效但概念简单的结构
    - 金字塔的底部是待处理图像的高分辨率表示，而顶部是低分辨率的近似，层级越高，图像越小，分辨率越低
    
    ```python
    cv2.pyrUp(image) # 对图像进行上采样
    cv2.pyrDown(image) # 对图像进行下采样
    ```
    
- 案例演示：[**geometric_transformation.ipynb**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/03OpenCV%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/geometric_transformation.ipynb)

# 1.2形态学操作

- 连通性
    - 在图像中，最小的单位是像素，每个像素周围有 8 个邻接像素，常见的邻接关系有 3 种
        - **4 邻接**：像素 $p(x, y)$ 的 4 邻域是：$(x+1, y); (x-1, y); (x, y+1); (x, y-1)$，用 $N_4(P)$ 表示像素 $p$ 的 4 邻接
        - **8 邻接**：像素 $p(x, y)$ 的 8 邻域是：4 邻域的点 + D 邻域的点，用 $N_8(p)$ 表示像素 p 的 8 邻域
        - **D 邻接**：像素 $p(x, y)$ 的 D 邻域是：$(x+1, y+1); (x-1, y-1); (x-1, y+1); (x+1, y-1)$，用 $N_D(P)$ 表示像素 $p$ 的 $D$ 邻接
    - 连通性是描述区域和边界的重要概念，两个像素连通的两个必要条件是
        1. 两个像素的位置是否相邻
        2. 两个像素的灰度值是否满足特定的相似性准则
    - 根据连通性的定义，有 4 联通、8 联通和 $m$ 联通三种
        - 4 连通：对于具有值 $V$ 的像素 $p$ 和 $q$，如果 $q$ 在集合 $N_4(p)$ 中，则称这两个像素是 4 连通
        - 8 连通：对于具有值 $V$ 的像素 $p$ 和 $q$，如果 $q$ 在集合 $N_8(p)$ 中，则称这两个像素是 8 连通
        - $m$ 连通：对于具有值 $V$ 的像素 $p$ 和 $q$，如果 $q$ 在集合 $N_4(p)$ 中**或** $q$ 在集合 $N_D(p)$ 中，并且 $N_4(P)$ 与 $N_4(g)$ 的交集为空（没有值 $V$ 的像素），则称这两个像素是 $m$ 连通的，即 4 连通和 $D$ 连通的混合连通
- 腐蚀和膨胀
    - 腐蚀：时原图中的高亮区域被蚕食，效果图拥有比原图更小的高亮区域，是求局部最小值的操作
    - 膨胀：使图像中高亮部分扩张，效果图拥有比原图更大的高亮区域，是求局部最大值的操作
    - 腐蚀
        - 用一个结构元素扫描图像中的每一个像素，用结构元素中的每一个像素与其覆盖的像素做“**与**”操作，如果都为 1，则该像素为 1，否则为 0
        
        ```python
        cv2.erode(image, kernel, iterations)
        ```
        
        - 参数
            - image：要处理的图像
            - kernel：核结构
            - iterations：腐蚀的次数，默认是 1
    - 膨胀
        - 用一个结构元素扫描图像中的每一个像素，用结构元素中的每一个像素与其覆盖的像素做“**与**”操作，如果都为 0，则该像素为 0，否则为 1
        
        ```python
        cv2.dilate(image, kernel, iterations)
        ```
        
        - 参数
            - image：要处理的图像
            - kernel：核结构
            - iterations：腐蚀的次数，默认是 1
- 开闭运算
    - 开运算
        - 开运算是先腐蚀后膨胀
        - 作用：分离物体，消除小区域
        - 特点：消除噪点，去除小的干扰块，而不影响原来的图像
    - 闭运算
        - 与开运算相反，是先膨胀后腐蚀
        - 作用：是消除/闭合物体里面的孔洞
        - 特点：可以填充闭合区域
    
    ```python
    cv2.morphologyEx(image, op, kernel)
    ```
    
    - 参数
        - img：要处理的图像
        - op：处理方式，开运算 `cv.MORPH_OPEN`，闭运算 `cv.MORPH_CLOSE`
        - kernel：核结构
- 礼帽和黑帽
    - 礼帽运算
        - 原图像与 “开运算“ 的结果图之差
        - 数学表达式：$dst = tophat(src, element) = src - open(src, element)$
        - 作用：
            - 用来分离比邻近点亮一些的斑块
            - 当一幅图像具有大幅的背景的时候，而微小物品比较有规律的情况下，可以使用顶帽运算进行背景提取
    - 黑帽运算
        - ”闭运算“ 的结果图与原图像之差
        - 数学表达式：$dst = blackhat(src, element) = close(src, element) - src$
        - 作用：用来分离比邻近点暗一些的斑块
    
    ```python
    cv2.morphologyEx(image, op, kernel)
    ```
    
    - 参数
        - image：要处理的图像
        - op：处理方式
            
            
            | 参数 | 功能 |
            | --- | --- |
            | cv.MORPH_CLOSE | 闭运算 |
            | cv.MORPH_OPEN | 开运算 |
            | cv.MORPH_TOPHAT | 礼帽运算 |
            | cv.MORPH_BLACKHAT | 黑帽运算 |
        - kernel：核结构
- 案例演示：[**morphological_operations.ipynb**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/03OpenCV%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/morphological_operations.ipynb)

# 1.3图像平滑

- 图像噪声
    - 椒盐噪声（脉冲噪声）
        - 是一种随机出现的白点或者黑点，可能是亮的区域有黑色像素或是在暗的区域有白色像素（或是两者皆有）
        - 成因：影像讯号受到突如其来的强烈干扰而产生、类比数位转换器或位元传输错误等
            - 例如
                - 失效的感应器导致像素值为最小值
                - 饱和的感应器导致像素值为最大值
    - 高斯噪声（正态噪声）
        - 是指噪声密度函数服从高斯分布的一类噪声
        - 高斯随机变量 $z$ 的概率密度函数：$p(z) = \frac{1}{\sqrt{2\pi}σ} e^{\frac{-(z - μ)^2}{2σ^2}}$
            - 其中，$z$ 表示灰度值， 表示 $z$ 的平均值或期望值，$σ$ 表示 $z$ 的标准差；标准差的平方 $σ^2$ 称为 $z$ 的方差
- 滤波器
    - 均值滤波
        - 采用均值滤波模板对图像噪声进行滤除
        - 令 $S_{xy}$ 表示中心在 $(x, y)$ 点，尺寸为 $m×n$ 的矩形子图像窗口的坐标
        组
        - 数学表达式：$\hat{f}(x, y) = \frac{1}{mn}\sum_{(s, t) ∈ S_{xy}}g(s, t)$
        - 优点：算法简单，计算速度较快
        - 缺点：去噪的同时去除了很多细节部分，将图像变得模糊
        
        ```python
        cv2.blur(src, ksize, anchor, borderType)
        ```
        
        - 参数
            - src：输入图像
            - ksize：卷积核的大小
            - anchor：默认值 $(-1, -1)$，表示核中心
            - borderType：边界类型
    - 高斯滤波
        - 二维高斯是构建高斯滤波器的基础
        - 概率分布函数：$G(x, y) = \frac{1}{2\piσ^2}exp\{-\frac{x^2 + y^2}{2σ^2}\}$
        - 高斯平滑在从图像中去除**高斯噪声**方面非常有效
        - 高斯平滑的流程
            1. 确定权重矩阵
            2. 计算高斯模糊
        
        ```python
        cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
        ```
        
        - 参数
            - src：输入图像
            - ksize：高斯卷积核的大小
                - **注意：卷积核的宽度和高度都应为奇数，且可以不同**
            - sigmaX：水平方向的标准差
            - sigmaY：垂直方向的标准差，默认值为 0，表示与 sigmaX 相同
            - borderType：填充边界类型
    - 中值滤波
        - 中值滤波是一种典型的非线性滤波技术，基本思想是用像素点邻域灰度值的中值来代替该像素点的灰度值
        - 中值滤波对**椒盐噪声**来说尤其有用，因为它不依赖于邻域内那些与典型值差别很大的值
        
        ```python
        cv2.medianBlur(src, ksize)
        ```
        
        - 参数
            - src：输入图像
            - ksize：卷积核的大小
- 案例演示：[**image_smoothing.ipynb**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/03OpenCV%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/image_smoothing.ipynb)

# 1.4直方图

- 灰度直方图
    - 原理
        - 直方图是对数据进行统计的一种方法，并且将统计值组织到一系列实现定义好的 bin 当中
            - 其中，bin 为直方图中经常用到的一个概念，可以译为**直条**或**组距**，其数值是从数据中计算出的特征统计量，这些数据可以是诸如梯度、方向、色彩或任何其他特征
        - **注意：直方图是根据灰度图进行绘制的，而不是彩色图像**
        - 直方图的一些**术语**和**细节**：
            - dims：需要统计的特征数目
            - bins：每个特征空间子区段的数目，可译为**直条**或**组距**
            - range：要统计特征的取值范围
        - 直方图的**意义**：
            - 直方图是图像中像素强度分布的图形表达方式
            - 它统计了每一个强度值所具有的像素个数
            - 不同的图像的直方图可能是相同的
    - 直方图的计算和绘制
        - 使用 OpenCV 中的方法统计直方图，并使用 matplotlib 将其绘制出来
        
        ```python
        cv2.calcHist([image], [channel], mask, [histSize], [range])
        ```
        
        - image：原图像，当传入函数时应该用中括号口括起来
        - channel：通道
            - 如果输入图像是灰度图，它的值就是[1]
            - 如果是彩色图像的话，传入的参数可以是 [0], [1], [2] 它们分别对应着通道 B, G, R
        - mask：掩模图像
        - histSize：bin 的数目
        - range：像素值范围，通常为 [0, 256]
- 掩膜的应用
    - 掩膜是用选定的图像、图形或物体，对要处理的图像进行挡，来控制图像处理的区域
    - 用途
        - 提取感兴趣区域
        - 屏蔽作用
        - 结构特征提取
        - 特殊形状图像制作
- 直方图均衡化
    - 原理：把原始图像的灰度直方图从比较集中的某个灰度区间变成在更广泛灰度范围内的分布
    - 应用：扩大图像像素值的分布范围，提高图像的对比度
    
    ```python
    cv2.equalizeHist(image)
    ```
    
    - 参数
        - image：灰度图像
- 自适应的直方图均衡化
    - 对整个图像进行直方图均衡化效果并不好
    - 自适应的直方图均衡化：整幅图像被分成很多小块，这些小块被称为 **tiles**（在 OpenCV 中 **tiles** 的大小默认是 8 × 8），然后再对每一个小块分别进行直方图均衡化，最后为了去除每一个小块之间的边界，再使用双线性差值对每一小块进行拼接
    
    ```python
    cl = cv2.createCLAHE(clipLimit, tileGridSize)
    result = cl.apply(image)
    ```
    
    - createCLAHE 参数
        - clipLimit：对比度限制，默认是 40
        - tileGridSize：分块的大小，默认为 8 × 8
    - apply 参数
        - image：灰度图像
- 案例演示：[**histogram.ipynb**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/03OpenCV%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/histogram.ipynb)

# 1.5边缘检测

- 概述
    - 目的：标识数字图像中亮度变化明显的点
    - 作用：大幅度地减少了数据量，并且剔除了可以认为不相关的信息，保留了图像重要的结构属性
    - 边缘检测方法
        - 基于搜索
            - 通过寻找图像**一阶导数**中的最大值来检测边界，然后利用计算结果估计边缘的局部方向，通常采用梯度的方向，并利用此方向找到局部梯度模的最大值
            - 代表算法：Sobel 算子和 Scharr 算子
        - 基于零穿越
            - 通过寻找图像**二阶导数**零穿越来寻找边界
            - 代表算法：Laplacian 算子
- Sobel 算子
    - Sobel 边缘检测算法比较简单，实际应用中效率比 Canny **效率更高**，但是边缘不如Canny 检测的准确
    - 注意：当内核大小为 3 时，Sobel 内核可能产生比较明显的误差，为解决这一问题，我们使用 Scharr 函数，但该函数仅作用于大小为 3 的内核
    - Scharr 函数该函数的运算与 Sobel 函数**一样快**，但结果却**更加精确**
    
    ```python
    cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)
    ```
    
    - 参数
        - src：传入的图像
        - ddepth：图像的深度
        - dx 和 dy：指求导的阶数，0 表示这个方向上没有求导，取值为0、1
        - ksize：是 Sobel 算子的大小，即卷积核的大小，必须为奇数 1、3、5、7，默认为 3
            - **注意：如果 ksize=-1，就演变成为 3 × 3 的 Scharr 算子**
        - scale：缩放导数的比例常数，默认情况为没有伸缩系数
        - borderType：图像边界的模式，默认值为 cv2.BORDER_DEFAULT
    
    ```python
    Scale_abs = cv2.convertScaleAbs(x) # 格式转换
    result = cv2.addWeighted(src1, alpha, src2, beta) # 图像混合
    ```
    
- Laplacian 算子
    - Laplacian 是利用二阶导数来检测边缘
    
    ```python
    cv2.Laplacian(src, ddepth, ksize)
    ```
    
    - 参数
        - src：需要处理的图像
        - ddepth：图像的深度，-1 表示采用的是原图像相同的深度，目标图像的深度必须大于等于原图像的深度
        - ksize：算子的大小，即卷积核的大小，必须为 1, 3, 5, 7
- Canny 边缘检测
    - Canny 边缘检测算法是一种非常流行的边缘检测算法，是 John F. Canny 于 1986 年提出的，被认为是最优的边缘检测算法
    - 原理
        1. 噪声去除：高斯滤波
        2. 计算图像梯度：sobel 算子，计算梯度大小和方向
        3. 非极大值抑制：利用梯度方向像素来判断当前像素是否为边界点
        4. 滞后阈值：设置两个阈值，确定最终的边界
    
    ```python
    cv2.Canny(image, threshold1, threshold2)
    ```
    
    - 参数
        - image：灰度图
        - threshold1：minval，较小的阈值将间断的边缘连接起来
        - threshold2：maxval，较大的阈值检测图像中明显的边缘
- 案例演示：[**edge_detection.ipynb**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/03OpenCV%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/edge_detection.ipynb)

# 1.6模板匹配

- 在给定的图片中查找和模板最相似的区域，该算法的输入包括模板和图片，整个任务的思路就是按照滑窗的思路不断的移动模板图片，计算其与图像中对应区域的匹配度，最终将匹配度最高的区域选择为最终的结果

```python
cv2.matchTemplate(image, template, method)
```

- 参数
    - img：要进行模板匹配的图像
    - template：模板
    - method：实现模板匹配的算法
        1. 平方差匹配(`cv.TM_SQDIFF`)：利用模板与图像之间的平方差进行匹配，最好的匹配是 0，匹配越差，匹配的值越大
        2. 相关匹配(`cv.TM_CCORR`)：利用模板与图像间的乘法进行匹配，数值越大表示匹配程度较高，越小表示匹配效果差
        3. 利用相关系数匹配(`cv.TM_CCOEFF`)：利用模板与图像间的相关系数匹配，1 表示完美的匹配，-1 表示最差的匹配
- 完成匹配
    - 使用 `cv.minMaxLoc` 方法查找最大值所在的位置即可
    - 如果使用平方差作为比较方法，则最小值位置是最佳匹配位置
- 案例演示：[**template_matching.ipynb**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/03OpenCV%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/template_matching.ipynb)

# 1.7霍夫变换

- 霍夫变换常用来提取图像中的直线和圆等几何形状
- 霍夫线检测
    
    ```python
    cv2.HoughLines(image, rho, theta, threshold)
    ```
    
    - 参数
        - img：检测的图像，要求是二值化的图像，所以在调用霍夫变换之前首先要进行二值化，或者进行 Canny 边缘检测
        - rho：$ρ$ 的精确度
        - theta：$θ$ 的精确度
        - threshold：阈值，只有累加器中的值高于该阈值时才被认为是直线
- 霍夫圆检测
    - OpenCV 中使用**霍夫梯度法**进行圆形的检测
    - 霍夫梯度法将霍夫圆检测范围两个阶段
        1. 检测圆心
        2. 利用圆心推导出圆半径
    
    ```python
    cv2.HoughCircles(image, method, dp, minDist, param1=100, param2=100, minRadius=0, maxRadius=0)
    ```
    
    - 参数
        - image：输入图像，应输入灰度图像
        - method：使用霍夫变换圆检测的算法，参数为 `cv.HOUGH_GRADIENT`
        - dp：霍夫空间的分辦率，dp=1 时表示霍夫空间与输入图像空间的大小一致，dp=2 时霍夫空间是输入图像空间的一半，以此类推
        - minDist：圆心之间的最小距离，如果检测到的两个圆心之间距离小于该值，则认为它们是同一个圆心
        - param1：边缘检测时使用 Canny 算子的高阈值，低阈值是高阈值的一半
        - param2：检测圆心和确定半径时所共有的阈值
        - minRadius：所检测到的圆半径的最小值
        - maxRadius：所检测到的圆半径的最大值
- 案例演示：[**hough_transform.ipynb**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/03OpenCV%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/hough_transform.ipynb)
