import cv2 as cv
import numpy as np
from PIL import Image

def show(image):
    """OpenCV BGR 이미지를 PIL로 변환해 보여줍니다."""
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def rotate_img(img, angle, center):
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    return cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR)

import os
src_dir = "../../Downloads/--.v1i.yolov11/train/images"
dst_dir = "../../Downloads/--.v1i.yolov11/train/new_images"

for fname in os.listdir(src_dir):
    # print(fname)
    src_path = os.path.join(src_dir, fname)
    src = cv.imread(src_path)
    if src is None:
        print(f"{fname} 읽기 실패")
    # 3) (선택) 큰 이미지는 절반 크기로 축소
    flag = False
    if src.shape[0] > 800 or src.shape[1] > 800:
        src = cv.resize(src, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        flag = True
    h, w = src.shape[:2]
    
    # 4) 원 검출
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (0,0), 1)
    circles = cv.HoughCircles(
        blur, cv.HOUGH_GRADIENT, 1, 50,
        param1=300, param2=40,
        minRadius=20, maxRadius=120
    )
    if circles is None:
        print(f"ℹ️  {fname} 에서 원 검출 실패 → 빈 이미지 저장")
        # 원 하나도 없으면 전부 검정 이미지
        result = np.zeros_like(src)
    else:
        circles_ = np.round(circles[0]).astype(int)
        # 5) 전체 크기 마스크 생성 & 원만 흰색으로 채우기
        mask = np.zeros((h, w), dtype=np.uint8)
        for cx, cy, r in circles_:
            cv.circle(mask, (cx, cy), r, 255, thickness=-1)
        result = cv.bitwise_and(src, src, mask=mask)
        if flag:
            src = cv.resize(src, None, fx=2.0, fy=2.0, interpolation=cv.INTER_LINEAR)
            
    # 7) 결과 저장
    save_path = os.path.join(dst_dir, fname)
    cv.imwrite(save_path, result)
    print(f"✅  저장됨: {save_path}")