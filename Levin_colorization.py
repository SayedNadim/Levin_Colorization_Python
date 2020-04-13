import colorsys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy import sparse
from scipy.sparse import linalg


class Colorize_Levin():
    def __init__(self):
        super(Colorize_Levin, self).__init__()

    def yiq_rgb(self, y, i, q):
        """
        Takes Y, I and Q channels and returns converted R, G and B channels
        :param y: Y channel of the image
        :param i: I channel of the image
        :param q: Q channel of the image
        :return: R, G and B channels
        """
        r_raw = y + 0.948262 * i + 0.624013 * q
        g_raw = y - 0.276066 * i - 0.639810 * q
        b_raw = y - 1.105450 * i + 1.729860 * q
        r_raw[r_raw < 0] = 0
        r_raw[r_raw > 1] = 1
        g_raw[g_raw < 0] = 0
        g_raw[g_raw > 1] = 1
        b_raw[b_raw < 0] = 0
        b_raw[b_raw > 1] = 1
        return (r_raw, g_raw, b_raw)

    def image_preprocess(self, original, hinted_image):
        """
        Takes original as well as hinted image and performs colorspace conversion.
        :param original: Original grayscale image
        :param hinted_image: Grayscale image with user scrabbles
        :return: Difference image and YIQ (YUV) colorspace image
        """
        original = original.astype(float) / 255
        hinted_image = hinted_image.astype(float) / 255
        colorIm = abs(original - hinted_image).sum(2) > 0.01
        (Y, _, _) = colorsys.rgb_to_yiq(original[:, :, 0], original[:, :, 1], original[:, :, 2])
        (_, I, Q) = colorsys.rgb_to_yiq(hinted_image[:, :, 0], hinted_image[:, :, 1], hinted_image[:, :, 2])
        ntscIm = np.zeros(original.shape)
        ntscIm[:, :, 0] = Y
        ntscIm[:, :, 1] = I
        ntscIm[:, :, 2] = Q
        return colorIm, ntscIm

    def colorize(self, original, hinted_image):
        """
        Takes original image as well as hinted image as inputs and returns the colorized image.
        :param original: Original grayscale image
        :param hinted_image: Grayscale image with user scrabbles
        :return: Colorized image
        """
        colorIm, ntscIm = self.image_preprocess(original, hinted_image)
        n = ntscIm.shape[0]  # n = image height
        m = ntscIm.shape[1]  # m = image width
        imgSize = n * m
        indsM = np.arange(imgSize).reshape(n, m, order='F').copy()
        wd = 1
        row_inds = np.zeros(imgSize * (2 * wd + 1) ** 2, dtype=np.int64)
        col_inds = np.zeros(imgSize * (2 * wd + 1) ** 2, dtype=np.int64)
        vals = np.zeros(imgSize * (2 * wd + 1) ** 2)
        length = 0
        consts_len = 0
        for j in range(m):
            for i in range(n):
                if not colorIm[i, j]:
                    tlen = 0
                    gvals = np.zeros((2 * wd + 1) ** 2)
                    for ii in range(max(0, i - wd), min(i + wd + 1, n)):
                        for jj in range(max(0, j - wd), min(j + wd + 1, m)):
                            if ii != i or jj != j:
                                row_inds[length] = consts_len
                                col_inds[length] = indsM[ii, jj]
                                gvals[tlen] = ntscIm[ii, jj, 0]
                                length += 1
                                tlen += 1
                    t_vals = ntscIm[i, j, 0].copy()
                    gvals[tlen] = t_vals
                    c_var = np.mean(
                        (gvals[0:tlen + 1] - np.mean(gvals[0:tlen + 1])) ** 2)
                    csig = c_var * 0.6
                    mgv = min((gvals[0:tlen + 1] - t_vals) ** 2)
                    if csig < (-mgv / np.log(0.01)):
                        csig = -mgv / np.log(0.01)
                    if csig < 0.000002:
                        csig = 0.000002
                    gvals[0:tlen] = np.exp(-((gvals[0:tlen] - t_vals) ** 2) / csig)
                    gvals[0:tlen] = gvals[0:tlen] / np.sum(
                        gvals[0:tlen])
                    vals[length - tlen:length] = -gvals[0:tlen]
                row_inds[length] = consts_len
                col_inds[length] = indsM[i, j]
                vals[length] = 1
                length += 1
                consts_len += 1

        vals = vals[0:length]
        col_inds = col_inds[0:length]
        row_inds = row_inds[0:length]

        # Optimization
        A = sparse.csr_matrix((vals, (row_inds, col_inds)), (consts_len, imgSize))
        b = np.zeros((A.shape[0]))
        nI = np.zeros(ntscIm.shape)
        nI[:, :, 0] = ntscIm[:, :, 0]
        colorCopy = colorIm.reshape(imgSize, order='F').copy()
        lblInds = np.nonzero(colorCopy)
        for t in [1, 2]:
            curIm = ntscIm[:, :, t].reshape(imgSize, order='F').copy()
            b[lblInds] = curIm[lblInds]
            new_vals = linalg.spsolve(A, b)
            nI[:, :, t] = new_vals.reshape(n, m, order='F')
        (R, G, B) = self.yiq_rgb(nI[:, :, 0], nI[:, :, 1], nI[:, :, 2])
        RGB = np.zeros(nI.shape)
        RGB[:, :, 0] = R
        RGB[:, :, 1] = G
        RGB[:, :, 2] = B
        return RGB


if __name__ == '__main__':
    original = cv2.cvtColor(cv2.imread('baby.bmp'),
                            cv2.COLOR_BGR2RGB)  # CV2 reads image in BGR format. We need to convert it to RGB format
    hinted_image = cv2.cvtColor(cv2.imread('baby_marked.bmp'),
                                cv2.COLOR_BGR2RGB)  # CV2 reads image in BGR format. We need to convert it to RGB format
    colorize = Colorize_Levin()
    colorized_image = colorize.colorize(original, hinted_image)
    misc.imsave('baby_colorized.bmp', colorized_image, format='bmp')
    plt.imshow(colorized_image)
    plt.show()
