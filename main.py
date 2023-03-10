import numpy as np
import cv2
from collections import Counter, defaultdict


# 遗留物体检测
def abandon_detect():
    first_frame_path = r'./videos/desk.jpg'
    file_path = 0
    consecutive_frame = 2  # 设置连续帧数

    first_frame = cv2.imread(first_frame_path)
    # first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_frame = cv2.resize(first_frame, (640, 480))
    # first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # first_frame_blur = cv2.GaussianBlur(first_frame_gray, (21, 21), 0)
    cap = cv2.VideoCapture(file_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter('outputs/abandon_output.mp4', fourcc, fps, (W, H))

    track_temp = []
    track_master = []
    track_temp2 = []

    top_contour_dict = defaultdict(int)
    # obj_detected_dict = defaultdict(int)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        count += 1
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)
        # print(first_frame.shape)
        # print(frame.shape)
        frame_diff = cv2.absdiff(first_frame, frame)
        edged = cv2.Canny(frame_diff, 30,
                          150)  # Canny Edge Detection, any gradient between 30 and 150 are considered edges
        cv2.imshow('CannyEdgeDet', edged)
        cv2.waitKey(15)

        kernel = np.ones((5, 5), np.uint8)  # higher the kernel, eg (10,10), more will be eroded or dilated
        thresh = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)  # 形态学滤波
        # cv2.imshow('Morph_Close', thresh)
        # cv2.waitKey(15)
        # 根据边缘特征轮廓
        # contours    ： 轮廓  M*N  M是轮廓个数  N是每个轮廓的点
        # hierarchy   ： 轮廓等级关系 M*4
        (cnts, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(cnts)
        # print(hierarchy)

        new_cnts = []
        for c in cnts:
            '''
            求每个轮廓矩
            （1）空间矩
            零阶矩：m00
            一阶矩：m10, m01
            二阶矩：m20, m11, m02
            三阶矩：m30, m21, m12, m03
            （2）中心矩
            二阶中心矩：mu20, mu11, mu02
            三阶中心矩：mu30, mu21, mu12, mu03
            （3）归一化中心矩
            二阶Hu矩：nu20, nu11, nu02
            三阶Hu矩：nu30, nu21, nu12, nu03
            '''
            M = cv2.moments(c)

            if M['m00'] == 0:
                pass
            else:  # 求轮廓重心cx cy
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # 设置轮廓面积的范围
                if cv2.contourArea(c) < 2500 or cv2.contourArea(c) > 200000:
                    pass
                else:
                    new_cnts.append(c)
                    (x, y, w, h) = cv2.boundingRect(c)  # 计算轮廓的边界框
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    # cv2.imshow('11', frame)
                    # cv2.waitKey(15)
                    cxcy = cx + cy
                    track_temp.append([cxcy, count])
                    track_master.append([cxcy, count])
                    count_unique = set(j for i, j in track_master)

                    if len(count_unique) > consecutive_frame or False:
                        min_count = min(j for i, j in track_master)
                        for i, j in track_master:
                            if j != min_count:
                                track_temp2.append([i, j])

                        track_master = list(track_temp2)  # transfer to the master list
                        track_temp2 = []

                    count_cxcy = Counter(i for i, j in track_master)  # 重心位置计数
                    # print('count_cxcy', count_cxcy)

                    for i, j in count_cxcy.items():
                        if j >= consecutive_frame:
                            top_contour_dict[i] += 1
                    print(top_contour_dict)  # {488: 80, 489: 19, 502: 48, 503: 167, 504: 710})  计数：连续20帧位置不变的轮廓

                    if cxcy in top_contour_dict:
                        if top_contour_dict[cxcy] > 5:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        cv2.imshow('Abandoned Object Detection', frame)
        out.write(frame)
        if cv2.waitKey(12) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# 运动停止检测
def stop_detect():
    previous_frame = cv2.imread('videos/coal.jpg')
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    file_path = 'videos/coal4.mp4'
    cap = cv2.VideoCapture(file_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter('outputs/stop_output.mp4', fourcc, fps, (W, H))

    count = 0
    static_count = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if count % 5 == 0:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)

            frame_diff = cv2.absdiff(previous_gray, frame_gray)
            mean = np.mean(frame_diff)
            previous_gray = frame_gray.copy()
            if mean < 1.0:
                # print('mean:', mean)
                static_count += 1
            if static_count > 10:
                cv2.putText(frame, 'stop', (300, 400), 2, 4, (0, 0, 255), 5)
        cv2.imshow('window', frame)
        count += 1
        out.write(frame)
        if cv2.waitKey(12) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    abandon_detect()
    # stop_detect()
