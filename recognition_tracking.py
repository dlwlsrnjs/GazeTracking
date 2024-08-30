import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
import csv
import os
from gaze_tracking import GazeTracking

# ONNX 모델 로드
onnx_model_path = 'path_to_your_model.onnx'  # 실제 모델 경로로 교체하세요
onnx_session = ort.InferenceSession(onnx_model_path)
print("ONNX 모델이 로드되었습니다.")

# GazeTracking 초기화 및 웹캠 설정
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
print("웹캠이 초기화되었습니다.")

# CSV 파일 생성 및 헤더 추가
csv_file = 'gaze_data.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Left Pupil', 'Right Pupil', 'Image Path', 'Caption'])
print(f"CSV 파일 {csv_file}이(가) 생성되었습니다.")

# 데이터 수집 및 처리 반복
capture_count = 0
while True:
    start_time = datetime.now()
    gaze_data = []
    frames = []
    image_paths = []

    print("5초 동안의 시선 데이터와 이미지를 수집 중...")
    # 5초 동안의 시선 데이터와 이미지를 수집
    while (datetime.now() - start_time).seconds < 5:
        # 웹캠에서 새로운 프레임 가져오기
        ret, frame = webcam.read()
        if not ret:
            print("웹캠에서 프레임을 가져오지 못했습니다.")
            break

        # 시선 데이터 분석
        gaze.refresh(frame)

        # 시선 좌표 수집
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        print(f"수집된 시선 좌표 - 왼쪽: {left_pupil}, 오른쪽: {right_pupil}")

        # 시선 좌표와 프레임을 리스트에 추가
        gaze_data.append((left_pupil, right_pupil))
        frames.append(frame.copy())

        # 현재 프레임을 디스플레이 (실시간 보기 용도)
        annotated_frame = gaze.annotated_frame()

        # 시선 좌표 텍스트 추가
        text = ""
        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        cv2.putText(annotated_frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        cv2.putText(annotated_frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(annotated_frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        # Gaze Tracking 창에 표시
        cv2.imshow("Gaze Tracking", annotated_frame)

        # ESC 키를 누르면 조기 종료
        if cv2.waitKey(1) == 27:
            webcam.release()
            cv2.destroyAllWindows()
            print("사용자에 의해 프로그램이 종료되었습니다.")
            exit()

    print("5초 동안의 데이터 수집이 완료되었습니다.")

    # ONNX 모델에 데이터 전달
    try:
        gaze_data_np = np.array(gaze_data, dtype=np.float32)
        frames_np = np.array(frames, dtype=np.uint8)

        print(f"모델 입력 준비 - 시선 데이터: {gaze_data_np.shape}, 프레임 데이터: {frames_np.shape}")

        # 모델 입력 준비 (모델에 맞게 조정)
        inputs = {
            'gaze_data': gaze_data_np,  # 모델 입력 이름에 맞게 조정
            'frames': frames_np         # 모델 입력 이름에 맞게 조정
        }

        # 모델 실행하여 캡션 생성
        outputs = onnx_session.run(None, inputs)
        caption = outputs[0]  # 모델 출력 형식에 맞게 인덱스를 조정
        print(f"모델 출력 캡션: {caption}")

    except Exception as e:
        print(f"모델 실행 중 오류 발생: {e}")
        continue

    # 마지막 프레임을 저장하고 경로 기록
    capture_count += 1
    image_path = f'image_{capture_count}.png'
    cv2.imwrite(image_path, frames[-1])  # 마지막 프레임을 저장
    print(f"프레임이 {image_path}에 저장되었습니다.")
    image_paths.append(image_path)

    # CSV 파일에 데이터 저장
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, left_pupil, right_pupil, image_path, caption])
    print(f"CSV에 데이터가 저장되었습니다: {timestamp}, {left_pupil}, {right_pupil}, {image_path}, {caption}")

    # 생성된 캡처 이미지와 캡션을 표시
    captured_image = cv2.imread(image_path)
    captioned_frame = cv2.putText(captured_image.copy(), caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    combined_frame = np.hstack((frames[-1], captioned_frame))  # 왼쪽에는 시선 데이터, 오른쪽에는 캡처 및 캡션

    # Captured Image & Caption 창에 표시
    cv2.imshow("Captured Image & Caption", combined_frame)

webcam.release()
cv2.destroyAllWindows()
print("프로그램이 정상 종료되었습니다.")
