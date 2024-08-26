import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_meshgrid(width, height, rows, cols):
    x = np.linspace(0, width, cols + 1, dtype=np.float32)
    y = np.linspace(0, height, rows + 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def distort_meshgrid(xx, yy, distortion_func):
    distorted_xx = xx + distortion_func(xx, yy)[0]
    distorted_yy = yy + distortion_func(xx, yy)[1]
    return distorted_xx, distorted_yy


def apply_meshgrid_distortion(image, xx, yy, distorted_xx, distorted_yy):
    height, width = image.shape[:2]
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    for i in range(len(xx) - 1):
        for j in range(len(yy) - 1):
            src = np.array([[xx[i, j], yy[i, j]],
                            [xx[i, j + 1], yy[i, j + 1]],
                            [xx[i + 1, j], yy[i + 1, j]],
                            [xx[i + 1, j + 1], yy[i + 1, j + 1]]], dtype=np.float32)

            dst = np.array([[distorted_xx[i, j], distorted_yy[i, j]],
                            [distorted_xx[i, j + 1], distorted_yy[i, j + 1]],
                            [distorted_xx[i + 1, j], distorted_yy[i + 1, j]],
                            [distorted_xx[i + 1, j + 1], distorted_yy[i + 1, j + 1]]], dtype=np.float32)

            transform_matrix = cv2.getPerspectiveTransform(src, dst)
            for x in range(int(xx[i, j]), int(xx[i, j + 1])):
                for y in range(int(yy[i, j]), int(yy[i + 1, j])):
                    map_x[y, x] = (transform_matrix[0, 0] * x + transform_matrix[0, 1] * y + transform_matrix[0, 2]) / \
                                  (transform_matrix[2, 0] * x + transform_matrix[2, 1] * y + transform_matrix[2, 2])
                    map_y[y, x] = (transform_matrix[1, 0] * x + transform_matrix[1, 1] * y + transform_matrix[1, 2]) / \
                                  (transform_matrix[2, 0] * x + transform_matrix[2, 1] * y + transform_matrix[2, 2])

    distorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return distorted_image


def display_grid(image, xx, yy):
    grid_image = image.copy()
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            cv2.circle(grid_image, (int(xx[i, j]), int(yy[i, j])), 2, (255, 0, 0), -1)
            if i < xx.shape[0] - 1:
                cv2.line(grid_image, (int(xx[i, j]), int(yy[i, j])), (int(xx[i + 1, j]), int(yy[i + 1, j])),
                         (255, 0, 0), 1)
            if j < xx.shape[1] - 1:
                cv2.line(grid_image, (int(xx[i, j]), int(yy[i, j])), (int(xx[i, j + 1]), int(yy[i, j + 1])),
                         (255, 0, 0), 1)
    return grid_image


# 读取图像
image = cv2.imread('C:/Users/admin/Desktop/321.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取图像的高度和宽度
height, width = image.shape[:2]

# 创建网格
rows, cols = 10, 10
xx, yy = create_meshgrid(width, height, rows, cols)


# 定义一个简单的变形函数
def distortion_func(xx, yy):
    return (np.sin(yy / 5) * 10, np.sin(xx / 5) * 10)


# 扭曲网格
distorted_xx, distorted_yy = distort_meshgrid(xx, yy, distortion_func)

# 应用网格变形
distorted_image = apply_meshgrid_distortion(image, xx, yy, distorted_xx, distorted_yy)

# 显示原始网格和扭曲网格
original_grid_image = display_grid(image, xx, yy)
distorted_grid_image = display_grid(distorted_image, distorted_xx, distorted_yy)

# 显示结果
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)

plt.imshow(original_grid_image)

plt.subplot(1, 2, 2)

plt.imshow(distorted_grid_image)

plt.show()
