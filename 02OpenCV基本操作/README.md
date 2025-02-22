# 02OpenCV基本操作

参考课程：

**【*黑马程序员* OpenCV入门教程】**

[https://www.bilibili.com/video/BV1Fo4y1d7JL]

@ZZHow(ZZHow1024)

# 2.1图像的基础操作

- 读取图像
    
    ```python
    cv2.imread(path, flag)
    ```
    
    - 参数
        - 要读取的图像路径
        - 读取方式的标志
            - cv.IMREAD_COLOR：以彩色模式加载图像，任何图像的透明度都将被忽略。（默认参数）(-1)
            - cv.IMREAD_GRAYSCALE：以灰度模式加载图像(0)
            - cv.IMREAD_UNCHANGED：包括 alpha 通道的加载图像模式(1)
    - **注意：如果加载的路径有错误，不会报错，会返回一个 None 值**
- 显示图像
    
    ```python
    cv2.imshow(title, image) # 通过 OpenCV 显示
    matplotlib.pyplot.imshow(image[:, :, ::-1]) # 通过 matplotlib 显示
    ```
    
    - 参数
        - 显示图像的窗口名称，以字符串类型表示
        - 要加载的图像
    - **注意：在调用 OpenCV 的显示图像的 API 后，要调用 cv.waitKey 给图像绘制留下时间**
- 案例：以灰度模式读取图像，分别用 OpenCV 和 matplotlib 的 API 显示图像，最后保存灰度图像
    
    ```python
    import cv2 as cv
    import matplotlib.pyplot as plt
    
    image = cv.imread("image.jpg", cv.IMREAD_GRAYSCALE)
    
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    plt.imshow(image, cmap="gray")
    plt.show()
    
    cv.imwrite("image_gray.jpg", image)
    ```
    
    - 案例演示：[**image_io.py**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/02OpenCV%E5%9F%BA%E6%9C%AC%E6%93%8D%E4%BD%9C/image_io.py)
- 绘制几何图形
    - 绘制直线
        
        ```python
        cv2.line(image, start, end, color, thickness)
        ```
        
        - 参数
            - image：要绘制直线的图像
            - start, end：直线的起点和终点
            - color：线条的颜色
            - thickness：线条宽度
    - 绘制矩形
        
        ```python
        cv2.rectangle(image, leftupper, rightdown, color, thickness)
        ```
        
        - 参数
            - image：要绘制矩形的图像
            - leftupper, rightdown：矩形的左上角和右下角坐标
            - color：线条的颜色
            - thickness：线条宽度
    - 绘制圆形
        
        ```python
        cv2.circle(image, centerpoint, r, color, thickness)
        ```
        
        - 参数
            - img：要绘制圆形的图像
            - centerpoint, r：圆心和半径
            - color：线条的颜色
            - thickness：线条宽度，为 -1 时生成闭合图案并填充颜色
    - 向图像中添加文字
        
        ```python
        cv2.putText(image, text, station, font, fontsize, color, thickness, linetype))
        ```
        
        - 参数
            - image：图像
            - text：要写入的文本数据
            - station：文本的放置位置
            - font：字体
            - fontsize：字体大小
            - color：文本颜色
            - thickness：线条宽度
            - linetype：`LINE_8`（默认）、`LINE_4`、`LINE_AA`
- 案例：生成一个全黑的图像，然后在里面绘制图像（直线、矩形 和 圆形）并添加文字（OpenCV）
    
    ```python
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt
    
    image = np.zeros((512, 512, 3), np.uint8)
    
    cv.line(image, (0, 0), (511, 511), (0, 0, 255), 3)
    cv.rectangle(image, (0, 0), (300, 300), (255, 0, 0), 3)
    cv.circle(image, (512 >> 1, 512 >> 1), 30, (0, 255, 0), -1)
    cv.putText(image, 'OpenCV', (100, 200), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv.LINE_AA)
    
    plt.imshow(image[:, :, ::-1])
    plt.title('OpenCV 绘制图形')
    plt.show()
    ```
    
    - 案例演示：[**draw_graphics.py**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/02OpenCV%E5%9F%BA%E6%9C%AC%E6%93%8D%E4%BD%9C/draw_graphics.py)
- 获取并修改图像中的像素点
    - 可以通过行和列的坐标值获取该像素点的像素值
        - 对于 BGR 图像，它返回一个蓝，绿，红值的数组
        - 对于灰度图像，仅返回相应的强度值
    
    ```python
    px = image[10, 10] # 获取某个像素点的像素值
    blue = image[100, 100, 0] # 仅获取蓝色通道的强度值
    image[100, 100] = [255, 255, 255] # 修改某个位置的像素值
    ```
    
    - 案例演示：[**modify_pixels.ipynb**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/02OpenCV%E5%9F%BA%E6%9C%AC%E6%93%8D%E4%BD%9C/modify_pixels.ipynb)
- 获取图像的属性
    - 图像属性包括**形状（行数**、**列数** 和 **通道数）**、**数据类型**、**大小**（**像素数**）等
        
        
        | 属性 | API |
        | --- | --- |
        | 形状 | image.shape |
        | 数据类型 | image.dtype |
        | 图像大小 | image.size |
    - 案例演示：[**get_image_properties.ipynb**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/02OpenCV%E5%9F%BA%E6%9C%AC%E6%93%8D%E4%BD%9C/get_image_properties.ipynb)
- 图像通道的拆分与合并
    - 有时需要将 BGR 图像分割为单个通道
    - 有时需要将单独的通道合并成 BGR 图像
    
    ```python
    b, g, r = cv.split(image) # 通道拆分
    image = cv.merge((b, g, r)) # 通道合并
    ```
    
- 色彩空间的改变
    - OpenCV 中有 150 多种颜色空间转换方法
    - 最广泛使用的转换方法有两种，**BGR → Gray** 和 **BGR → HSV**
    
    ```python
    cv.cvtColor(image, flag)
    ```
    
    - 参数
        - image：进行颜色空间转换的图像
        - flag：转换类型
            - cv.COLOR_BGR2GRAY：**BGR → Gray**
            - cv.COLOR_BGR2HSV：**BGR → HSV**

# 2.2算数操作

- 图像的加法
    - 可以使用 OpenCV 的 `cv2.add()` 函数把两幅图像相加，或者可以简单地通过 NumPy 操作添加两个图像，如 `res = image1 +image2`
    - 两个图像应该具有相同的大小和类型，或者第二个图像可以是标量值
    - 注意：OpenCV 加法和 NumPy 加法之间存在差异
        - OpenCV 的加法是**饱和操作**
        - NumPy 的加法是**模运算**
- 图像的混合
    - 其实也是加法，但是两幅图像的权重不同，这就会给人一种**混合**或者**透明**的感觉
    - 图像混合的计算公式：
        - $g(x) = (1 - \alpha) f_0(x) + \alpha f_1(x)$
        - 通过修改 $\alpha$ 的值 (0~1)，可以实现非常炫酷的混合
    - 将两幅图混合在一起
        - 第一幅图的权重是 $\alpha$，第二幅图的权重是 $\beta$
        - 函数 `cv2.addWeighted()` 可以按下面的公式对图片进行混合操作：
            - $dist = \alpha ⋅ image_1 + \beta ⋅ image_2 + \gamma$
- 案例演示：[**arithmetic_operations.ipynb**](https://github.com/ZZHow1024/OpenCV-Learning/blob/main/02OpenCV%E5%9F%BA%E6%9C%AC%E6%93%8D%E4%BD%9C/arithmetic_operations.ipynb)
