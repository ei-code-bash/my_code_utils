import cv2
import numpy as np
def fray_img(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def harris_detect(image_path):
    img=cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Harris角点检测
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # 结果优化
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 标记角点为红色

    return img


# 安装: pip install mediapipe
import mediapipe as mp


def mediapipe_face_detection(image_path):
    # 初始化MediaPipe面部网格
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5)

    # 读取图像
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 处理图像
    results = face_mesh.process(rgb_img)

    # 绘制关键点
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    return img
def main():
    image_path='C:/Users/30801/PycharmProjects/ooo/model/VCG41N1435038051.jpg'
    img_test = harris_detect(image_path)
    img_gray=fray_img(image_path)
    img_face=mediapipe_face_detection(image_path)
    cv2.imshow('gray', img_gray)
    cv2.imshow('test', img_test)
    cv2.imshow('face', img_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()