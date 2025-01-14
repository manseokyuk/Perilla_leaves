import os
import cv2  # OpenCV 라이브러리 사용
from ultralytics import YOLO

## YOLOv8 모델 불러오기 (사전 학습된 모델 사용)
# yolov8n.pt: Nano - 빠르지만 성능이 낮음.
# yolov8s.pt: Small - 빠르고 성능이 중간 정도.
# yolov8m.pt: Medium - 속도와 성능의 균형.
# yolov8l.pt: Large - 성능이 좋지만 느림.
# yolov8x.pt: XLarge - 최고 성능, 속도는 가장 느림.
model = YOLO('yolov8n.pt')

# 예측할 이미지 및 저장할 경로
image_dir = 'E:/Python_Project/data/label_image/'  # 예측할 이미지가 있는 디렉토리
save_dir = 'E:/Python_Project/data/results'  # 결과 이미지를 저장할 디렉토리

# 디렉토리 내 모든 이미지 파일 가져오기
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 'large' 클래스의 인덱스
large_class_name = 'large'  # YOLOv8에서 클래스명을 기준으로 필터링

# 각 이미지에 대해 예측 수행 (YOLOv8)
for image_file in image_files:
    img_path = os.path.join(image_dir, image_file)  # 이미지 전체 경로
    results = model(img_path)  # YOLOv8 예측 수행

    # YOLOv8 결과에서 'large' 클래스만 필터링
    large_results = [r for r in results[0].boxes if model.names[int(r.cls)] == large_class_name]

    # 필터링된 결과 시각화 및 저장 (YOLOv8)
    if len(large_results) > 0:  # 'large' 클래스가 존재할 경우
        img = cv2.imread(img_path)  # 원본 이미지 읽기
        for box in large_results:  # 필터링된 결과 반복
            xyxy = box.xyxy[0]  # 경계 상자 좌표
            conf = box.conf[0]  # 신뢰도
            # 경계 상자 그리기
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
            # 클래스명과 신뢰도 표시
            label = f"{large_class_name}: {conf:.2f}"  # 클래스명과 신뢰도 텍스트
            cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 결과를 지정된 경로에 저장
        save_path = os.path.join(save_dir, image_file)  # 저장할 파일 경로
        cv2.imwrite(save_path, img)  # 이미지 저장

# YOLOv8 결과 시각화 및 저장
for image_file in image_files:
    img_path = os.path.join(image_dir, image_file)  # 이미지 전체 경로
    results = model(img_path)  # YOLOv8 예측 수행

    # YOLOv8 전체 결과 시각화 및 저장
    results.show()  # 예측 결과 시각화
    results.save(save_dir="runs/detect")  # 결과 저장할 디렉토리 설정