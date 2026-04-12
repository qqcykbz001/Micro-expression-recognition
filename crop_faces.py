import os
import cv2
import numpy as np
from tqdm import tqdm

# 配置
INPUT_DIR = 'SAMM/SAMM'
OUTPUT_DIR = 'SAMM/Cropped'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化OpenCV的人脸检测器
print("初始化人脸检测器...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 人脸裁剪函数
def crop_face(image_path, output_path, margin=10, min_face_size=30, scale_factor=1.1, min_neighbors=5):
    """裁剪图像中的人脸并保存
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        margin: 人脸边界扩展像素数（默认10）
        min_face_size: 最小人脸大小（默认30）
        scale_factor: 检测缩放因子（默认1.1）
        min_neighbors: 最小邻居数（默认5）
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return False
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_face_size, min_face_size))
    
    if len(faces) == 0:
        print(f"未检测到人脸: {image_path}")
        # 保存原始图像
        cv2.imwrite(output_path, img)
        return False
    
    # 获取最大的人脸
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # 提取人脸区域
    x, y, w, h = largest_face
    
    # 扩展边界以包含更多上下文
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2 * margin)
    h = min(img.shape[0] - y, h + 2 * margin)
    
    # 裁剪人脸
    face_crop = img[y:y+h, x:x+w]
    
    # 保存裁剪后的图像
    cv2.imwrite(output_path, face_crop)
    return True

# 遍历所有图像
def process_dataset(margin=10, min_face_size=30, scale_factor=1.1, min_neighbors=5):
    """处理整个数据集
    
    Args:
        margin: 人脸边界扩展像素数（默认10）
        min_face_size: 最小人脸大小（默认30）
        scale_factor: 检测缩放因子（默认1.1）
        min_neighbors: 最小邻居数（默认5）
    """
    total_images = 0
    processed_images = 0
    
    # 遍历受试者目录
    for subject in tqdm(sorted(os.listdir(INPUT_DIR)), desc="处理受试者"):
        subject_path = os.path.join(INPUT_DIR, subject)
        if not os.path.isdir(subject_path):
            continue
        
        # 创建输出目录
        output_subject_path = os.path.join(OUTPUT_DIR, subject)
        os.makedirs(output_subject_path, exist_ok=True)
        
        # 遍历视频目录
        for video in sorted(os.listdir(subject_path)):
            video_path = os.path.join(subject_path, video)
            if not os.path.isdir(video_path):
                continue
            
            # 创建输出目录
            output_video_path = os.path.join(output_subject_path, video)
            os.makedirs(output_video_path, exist_ok=True)
            
            # 遍历图像文件
            for image_file in sorted(os.listdir(video_path)):
                if not image_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                total_images += 1
                input_image_path = os.path.join(video_path, image_file)
                output_image_path = os.path.join(output_video_path, image_file)
                
                if crop_face(input_image_path, output_image_path, margin=margin, min_face_size=min_face_size, scale_factor=scale_factor, min_neighbors=min_neighbors):
                    processed_images += 1
    
    print(f"\n处理完成！")
    print(f"总图像数: {total_images}")
    print(f"成功裁剪人脸: {processed_images}")
    print(f"未检测到人脸: {total_images - processed_images}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="裁剪数据集中的人脸")
    parser.add_argument('--margin', type=int, default=2, help='人脸边界扩展像素数（默认10）')
    parser.add_argument('--min-face-size', type=int, default=30, help='最小人脸大小（默认30）')
    parser.add_argument('--scale-factor', type=float, default=1.1, help='检测缩放因子（默认1.1）')
    parser.add_argument('--min-neighbors', type=int, default=5, help='最小邻居数（默认5）')
    
    args = parser.parse_args()
    
    print(f"使用参数: margin={args.margin}, min_face_size={args.min_face_size}, scale_factor={args.scale_factor}, min_neighbors={args.min_neighbors}")
    process_dataset(margin=args.margin, min_face_size=args.min_face_size, scale_factor=args.scale_factor, min_neighbors=args.min_neighbors)