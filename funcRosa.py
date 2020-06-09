import numpy as np
import qimage2ndarray

def edge_detection(image):
    image_array = qimage2ndarray.rgb_view(image)
    image_gray_array = np.dot(image_array,[0.2989, 0.5870, 0.1140])
    image_padding = np.pad(image_gray_array, 1, 'constant', constant_values=(0))

    image_pad = qimage2ndarray.array2qimage(image_padding, normalize=False)

    gaus_2D = getGKernel((51, 51), 13)
    image_2D = filtering(image_pad, gaus_2D)

    lap_image = Laplacian(image_2D)

    image_final = qimage2ndarray.array2qimage(lap_image, normalize=False)
    return QPixmap.fromImage(image_final)

def getGKernel(shape, sigma):
    # a = shape[0] , b = shape[1] , (s = 2a+1, t = 2b+1)
    s = (shape[0] - 1) / 2
    t = (shape[1] - 1) / 2

    y, x = np.ogrid[-s:s + 1, -t:t + 1]
    gaus_kernel = np.exp(-(x * x + y * y)) / (2. * sigma * sigma)
    sum = gaus_kernel.sum()
    gaus_kernel /= sum
    return gaus_kernel

def filtering(img, kernel, boundary=0):
    row, col = len(img), len(img[0])
    ksizeY, ksizeX = kernel.shape[0], kernel.shape[1]

    pad_image = my_padding(img, (ksizeY, ksizeX), boundary=boundary)
    filtered_img = np.zeros((row, col), dtype=np.float32)
    for i in range(row):
        for j in range(col):
            filtered_img[i, j] = np.sum(
                np.multiply(kernel, pad_image[i:i + ksizeY, j:j + ksizeX]))  # filter * image
    return filtered_img


def Laplacian(gaus_array):

    d = gaus_arr.shape
    n = d[0]
    m = d[1]
    lap_array = np.copy(gaus_array)

    for i in range(1, n - 1):
        for j in range(1, m - 1):
            lap = gaus_array[i - 1][j - 1] + gaus_array[i][j - 1] + gaus_array[i + 1][j - 1] + gaus_array[i - 1][j] + \
                  gaus_array[i][j] * (-8) + gaus_array[i + 1][j] + gaus_array[i - 1][j + 1] + gaus_array[i][j + 1] + \
                  gaus_array[i + 1][j + 1]

            lap_array[i][j] = lap

    return lap_array
