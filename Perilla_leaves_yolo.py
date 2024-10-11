import torch
import cv2
import os

# ##전체 데이터 확인

# # 학습된 모델 불러오기
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='E:/Python_Project/Perilla leaves_project/yolov5/runs/train/exp2/weights/best.pt')

# # 예측할 이미지 경로
# image_dir = 'E:/Python_Project/data/label_image/'  # 예측할 이미지가 있는 디렉토리

# # 디렉토리 내 모든 이미지 파일 가져오기
# image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# # 각 이미지에 대해 예측 수행
# for image_file in image_files:
#     img_path = os.path.join(image_dir, image_file)  # 이미지 전체 경로
#     results = model(img_path)  # 예측 수행

#     # 결과 출력 및 저장
#     results.show()  # 예측 결과 시각화
#     results.save()  # 예측 결과 저장





##특정 클래스(large)만 필더랑해서 표시

import torch
import os
import cv2  # OpenCV 라이브러리 사용

# 학습된 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'custom', path='E:/Python_Project/Perilla leaves_project/yolov5/runs/train/exp/weights/best.pt')

# 예측할 이미지 경로
image_dir = 'E:/Python_Project/data/label_image/'  # 예측할 이미지가 있는 디렉토리
save_dir = 'E:/Python_Project/data/results'  # 결과 이미지를 저장할 디렉토리

# 디렉토리 내 모든 이미지 파일 가져오기
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 'large' 클래스의 인덱스
large_class_index = 2  # 'large'는 0부터 시작하므로 2

# 각 이미지에 대해 예측 수행
for image_file in image_files:
    img_path = os.path.join(image_dir, image_file)  # 이미지 전체 경로
    results = model(img_path)  # 예측 수행

    # 결과에서 'large' 클래스만 필터링
    large_results = results.xyxy[0]  # 예측 결과의 좌표 가져오기
    large_results = large_results[large_results[:, -1] == large_class_index]  # 'large' 클래스 필터링

    # 필터링된 결과 시각화
    if len(large_results) > 0:  # 'large' 클래스가 존재할 경우
        # 결과를 업데이트하고 시각화
        img = cv2.imread(img_path)  # 원본 이미지 읽기
        for *xyxy, conf, cls in large_results:  # 필터링된 결과 반복
            # 경계 상자 그리기
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
            # 클래스명과 신뢰도 표시
            label = f"large: {conf:.2f}"  # 클래스명과 신뢰도 텍스트
            cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 결과를 지정된 경로에 저장
        save_path = os.path.join(save_dir, image_file)  # 저장할 파일 경로
        cv2.imwrite(save_path, img)  # 이미지 저장


