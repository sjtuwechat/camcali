import cv2
import numpy as np
import glob
import os


def calibrate():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.00001)
    Nx_cor = 9
    Ny_cor = 6

    objp = np.zeros((Nx_cor * Ny_cor, 3), np.float32)
    objp[:, :2] = np.mgrid[0:Nx_cor, 0:Ny_cor].T.reshape(-1, 2)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(r"img/*.jpg")
    # print(images)
    for fname in images:
        # print(image)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret是找到角点的flag，corners是角点
        ret, corners = cv2.findChessboardCorners(gray, (Nx_cor, Ny_cor), None)
        # print(corners)

        if ret == True:
            # 亚像素级角点检测
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, (Nx_cor, Ny_cor), corners2, ret)
            cv2.imshow('img', img)

            cv2.imshow('one', gray)
            cv2.waitKey(1000)
        # cv2.imshow('img',cv2.resize(img,(480,640)))
        else:
            print(fname)



    # mtx,相机内参；dist畸变系数；
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(mtx, dist)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error / len(objpoints))

    # 保存相机内参
    np.savez('calibrate.npz', mtx=mtx, dist=dist[0:4])

    return mtx, dist

# 利用内参去畸变
def undistortion(img, mtx, dist):
    # cv2.imshow('img', img)
    # cv2.waitKey(2000)
    h, w = img.shape[:2]
    # print(h, w)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # print(roi)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    # cv2.imwrite('calibresult.png', dst)
    # cv2.imshow('cali', dist)
    # cv2.waitKey(2000)
    return dst


def undistort_img_from_folder(path, save_path, npzfile_path):
    """
    对文件夹中的图片进行畸变校正
    :param path: 原图路径
    :param save_path: 矫正后的保存路径
    :param npzfile_path: 矫正矩阵
    :return: 无
    """
    try:
        npzfile = np.load(npzfile_path)
        mtx = npzfile['mtx']
        dist = npzfile['dist']
    except IOError:
        mtx, dist = calibrate()

    for root, dirs, files in os.walk(path):
        for filename in files:
            img = cv2.imread(root + filename)
            dist_img = undistortion(img, mtx, dist)
            cv2.imwrite(save_path + filename, dist_img)

    print("对原文件夹中的图像进行鱼眼矫正完成")

if __name__ == '__main__':

    mtx = []
    dist = []
    try:
        npzfile = np.load('calibrate02.npz')
        # npzfile = np.load('calibrate02.npz')
        mtx = npzfile['mtx']
        dist = npzfile['dist']
    except IOError:
        mtx, dist = calibrate()

    # print(dist[0:4])
    # print(dist)
    path = "/Users/yxlian/Desktop/multiCamfusion/fisheye/raw_data"
    save_path = "/Users/yxlian/Desktop/multiCamfusion/fisheye/output/"
    # path = '../extract_frames/data/p3_left'
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(".jpg"):
                # pass
                # print(os.path.join(root, filename))
                img = cv2.imread(os.path.join(root, filename))
                dist_img = undistortion(img, mtx, dist)
                print(os.path.join(save_path, filename))
                cv2.imwrite(os.path.join(save_path, filename), dist_img)
            # print(img)

    # img = cv2.imread('../mytest/left.jpg')
    # img = undistortion(img, mtx, dist)
    # cv2.imshow('img', img)
    # cv2.waitKey(1000)
    # cv2.imwrite('out_left.jpg', img)