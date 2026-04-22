import os
import cv2
import numpy as np
from tqdm import tqdm

# 配置
INPUT_DIR = 'CASME2/CASME2'
OUTPUT_DIR = 'CASME2/CASME2_Cropped'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 使用 OpenCV 内置的 Haar 级联分类器进行人脸检测（更准确）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 或者使用 DNN 模型进行人脸检测（可选）
# face_proto = "deploy.prototxt"
# face_model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
# face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)

# 人脸裁剪函数
def crop_face(image_path, output_path, margin=10, min_face_size=30, scale_factor=1.1, min_neighbors=5, use_dnn=False, tight_crop=False):
    """裁剪图像中的人脸并保存
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        margin: 人脸边界扩展像素数（默认10）
        min_face_size: 最小人脸大小（默认30）
        scale_factor: Haar级联检测器缩放比例（默认1.1）
        min_neighbors: Haar级联检测器邻居数（默认5）
        use_dnn: 是否使用DNN模型进行人脸检测（默认False）
        tight_crop: 是否进行紧密裁剪（聚焦于面部五官区域，默认False）
    """
    # 读取图像
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return False
    
    # 转换为灰度图用于人脸检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 存储检测到的人脸
    faces = []
    
    if use_dnn:
        # 使用 DNN 模型进行人脸检测（更精确）
        try:
            # 加载 DNN 人脸检测模型（如果文件存在）
            face_proto = "deploy.prototxt"
            face_model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
            if os.path.exists(face_proto) and os.path.exists(face_model):
                face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
                
                (h, w) = img.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                face_net.setInput(blob)
                detections = face_net.forward()
                
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    # 设置置信度阈值
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        # 确保边界在图像范围内
                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w, endX)
                        endY = min(h, endY)
                        
                        # 计算人脸尺寸
                        face_width = endX - startX
                        face_height = endY - startY
                        
                        if face_width >= min_face_size and face_height >= min_face_size:
                            faces.append((startX, startY, endX, endY, confidence))
        except Exception as e:
            print(f"DNN 人脸检测失败，回退到 Haar 级联: {str(e)}")
            # 如果 DNN 模型不可用，则回退到 Haar 级联
            faces = []
    
    # 如果未使用 DNN 或 DNN 失败，使用 Haar 级联检测器
    if not faces:
        faces_rect = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_face_size, min_face_size)
        )
        
        # 将检测结果转换为标准格式
        for (x, y, w, h) in faces_rect:
            faces.append((x, y, x+w, y+h, 1.0))  # 最后一个值是置信度
    
    # 计算每个人脸的面积，选择最大的人脸
    largest_face = None
    max_area = 0
    
    for (x, y, x_plus_w, y_plus_h, confidence) in faces:
        # 计算人脸面积
        width = x_plus_w - x
        height = y_plus_h - y
        area = width * height
        
        if area > max_area and min(width, height) >= min_face_size:
            max_area = area
            largest_face = (y, x_plus_w, y_plus_h, x)  # 转换为 (top, right, bottom, left) 格式
    
    if largest_face is None:
        print(f"未检测到足够大的人脸: {image_path}")
        cv2.imwrite(output_path, img)
        return False
    
    # 获取最大人脸的位置
    top, right, bottom, left = largest_face
    
    # 根据tight_crop参数决定扩展边界的方式
    if tight_crop:
        # 紧密裁剪：进一步调整边界以聚焦于五官
        face_height = bottom - top
        face_width = right - left
        
        # 计算人脸中心点
        center_x = left + face_width // 2
        center_y = top + face_height // 2
        
        # 调整边界以更好地聚焦五官（眼睛、鼻子、嘴巴区域）
        # 上边界移到眉毛上方一点点
        top = max(0, center_y - int(face_height * 0.30))
        # 下边界移到下巴上方
        bottom = min(img.shape[0], center_y + int(face_height * 0.50))
        # 左右边界以中心为基准，按比例收缩
        half_width = int(face_width * 0.43)
        left = max(0, center_x - half_width)
        right = min(img.shape[1], center_x + half_width)
        
        # 确保裁剪区域接近正方形
        crop_width = right - left
        crop_height = bottom - top
        target_size = min(crop_width, crop_height)  # 使用较小的尺寸作为正方形边长
        
        # 重新计算以创建正方形区域
        half_size = target_size // 2
        center_x_new = left + crop_width // 2
        center_y_new = top + crop_height // 2
        
        # 确保正方形区域在图像边界内
        top = max(0, center_y_new - half_size)
        bottom = min(img.shape[0], center_y_new + half_size)
        left = max(0, center_x_new - half_size)
        right = min(img.shape[1], center_x_new + half_size)
    elif margin > 0:
        # 减少边距以获得更紧凑的人脸区域
        extend_margin = int(margin/2)
        top = max(0, top - extend_margin)
        left = max(0, left - extend_margin)
        bottom = min(img.shape[0], bottom + extend_margin)
        right = min(img.shape[1], right + extend_margin)
        
        # 确保裁剪区域接近正方形
        crop_width = right - left
        crop_height = bottom - top
        target_size = min(crop_width, crop_height)  # 使用较小的尺寸作为正方形边长
        
        # 重新计算以创建正方形区域
        half_size = target_size // 2
        center_x_new = left + crop_width // 2
        center_y_new = top + crop_height // 2
        
        # 确保正方形区域在图像边界内
        top = max(0, center_y_new - half_size)
        bottom = min(img.shape[0], center_y_new + half_size)
        left = max(0, center_x_new - half_size)
        right = min(img.shape[1], center_x_new + half_size)
    
    # 裁剪人脸
    face_crop = img[top:bottom, left:right]
    
    # 检查裁剪后的图像是否有效
    if face_crop.size == 0:
        print(f"裁剪失败: {image_path}")
        cv2.imwrite(output_path, img)
        return False
    
    # 转换为灰度图
    if len(face_crop.shape) == 3:
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    # 保存裁剪后的图像
    cv2.imwrite(output_path, face_crop)
    return True

# 遍历所有图像
def process_dataset(margin=10, min_face_size=30, scale_factor=1.1, min_neighbors=5, use_dnn=False, tight_crop=False):
    """处理整个数据集
    
    Args:
        margin: 人脸边界扩展像素数（默认10）
        min_face_size: 最小人脸大小（默认30）
        scale_factor: Haar级联检测器缩放比例（默认1.1）
        min_neighbors: Haar级联检测器邻居数（默认5）
        use_dnn: 是否使用DNN模型进行人脸检测（默认False）
        tight_crop: 是否进行紧密裁剪（聚焦于面部五官区域，默认False）
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
            
            # 获取该视频目录下的所有图像文件
            image_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            # 存储人脸位置信息
            face_position = None
            
            # 遍历图像文件
            for i, image_file in enumerate(image_files):
                total_images += 1
                input_image_path = os.path.join(video_path, image_file)
                output_image_path = os.path.join(output_video_path, image_file)
                
                if i == 0:
                    # 处理第一个文件，检测人脸位置
                    # 先读取图像，检测人脸位置
                    img = cv2.imread(input_image_path)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # 检测人脸
                        faces = []
                        if use_dnn:
                            try:
                                face_proto = "deploy.prototxt"
                                face_model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
                                if os.path.exists(face_proto) and os.path.exists(face_model):
                                    face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
                                    (h, w) = img.shape[:2]
                                    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                                    face_net.setInput(blob)
                                    detections = face_net.forward()
                                    
                                    for j in range(0, detections.shape[2]):
                                        confidence = detections[0, 0, j, 2]
                                        if confidence > 0.5:
                                            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                                            (startX, startY, endX, endY) = box.astype("int")
                                            face_width = endX - startX
                                            face_height = endY - startY
                                            if face_width >= min_face_size and face_height >= min_face_size:
                                                faces.append((startX, startY, endX, endY, confidence))
                            except Exception as e:
                                # 回退到Haar级联
                                faces = []
                        
                        if not faces:
                            faces_rect = face_cascade.detectMultiScale(
                                gray,
                                scaleFactor=scale_factor,
                                minNeighbors=min_neighbors,
                                minSize=(min_face_size, min_face_size)
                            )
                            for (x, y, w, h) in faces_rect:
                                faces.append((x, y, x+w, y+h, 1.0))
                        
                        # 选择最大的人脸
                        largest_face = None
                        max_area = 0
                        for (x, y, x_plus_w, y_plus_h, confidence) in faces:
                            width = x_plus_w - x
                            height = y_plus_h - y
                            area = width * height
                            if area > max_area and min(width, height) >= min_face_size:
                                max_area = area
                                largest_face = (y, x_plus_w, y_plus_h, x)  # (top, right, bottom, left)
                        
                        if largest_face is not None:
                            top, right, bottom, left = largest_face
                            
                            # 应用裁剪逻辑，与crop_face函数一致
                            if tight_crop:
                                face_height = bottom - top
                                face_width = right - left
                                center_x = left + face_width // 2
                                center_y = top + face_height // 2
                                top = max(0, center_y - int(face_height * 0.40))
                                bottom = min(img.shape[0], center_y + int(face_height * 0.55))
                                half_width = int(face_width * 0.42)
                                left = max(0, center_x - half_width)
                                right = min(img.shape[1], center_x + half_width)
                                crop_width = right - left
                                crop_height = bottom - top
                                target_size = min(crop_width, crop_height)
                                half_size = target_size // 2
                                center_x_new = left + crop_width // 2
                                center_y_new = top + crop_height // 2
                                top = max(0, center_y_new - half_size)
                                bottom = min(img.shape[0], center_y_new + half_size)
                                left = max(0, center_x_new - half_size)
                                right = min(img.shape[1], center_x_new + half_size)
                            elif margin > 0:
                                extend_margin = int(margin/2)
                                top = max(0, top - extend_margin)
                                left = max(0, left - extend_margin)
                                bottom = min(img.shape[0], bottom + extend_margin)
                                right = min(img.shape[1], right + extend_margin)
                                crop_width = right - left
                                crop_height = bottom - top
                                target_size = min(crop_width, crop_height)
                                half_size = target_size // 2
                                center_x_new = left + crop_width // 2
                                center_y_new = top + crop_height // 2
                                top = max(0, center_y_new - half_size)
                                bottom = min(img.shape[0], center_y_new + half_size)
                                left = max(0, center_x_new - half_size)
                                right = min(img.shape[1], center_x_new + half_size)
                            
                            # 保存人脸位置
                            face_position = (top, right, bottom, left)
                            
                            # 使用计算出的人脸位置裁剪第一个文件
                            # 确保裁剪区域在图像边界内
                            top_clamp = max(0, top)
                            left_clamp = max(0, left)
                            bottom_clamp = min(img.shape[0], bottom)
                            right_clamp = min(img.shape[1], right)
                            
                            # 裁剪人脸
                            face_crop = img[top_clamp:bottom_clamp, left_clamp:right_clamp]
                            if face_crop.size > 0:
                                # 转换为灰度图
                                if len(face_crop.shape) == 3:
                                    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                                cv2.imwrite(output_image_path, face_crop)
                                processed_images += 1
                            else:
                                # 裁剪失败，保存原始图像（转换为灰度图）
                                if len(img.shape) == 3:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                cv2.imwrite(output_image_path, img)
                        else:
                            # 未检测到人脸，使用原始方法
                            if crop_face(input_image_path, output_image_path, margin=margin, min_face_size=min_face_size, 
                                       scale_factor=scale_factor, min_neighbors=min_neighbors, use_dnn=use_dnn, tight_crop=tight_crop):
                                processed_images += 1
                    else:
                        # 无法读取图像，使用原始方法
                        if crop_face(input_image_path, output_image_path, margin=margin, min_face_size=min_face_size, 
                                   scale_factor=scale_factor, min_neighbors=min_neighbors, use_dnn=use_dnn, tight_crop=tight_crop):
                            processed_images += 1
                else:
                    # 使用第一个文件的人脸位置裁剪
                    if face_position is not None:
                        top, right, bottom, left = face_position
                        img = cv2.imread(input_image_path)
                        if img is not None:
                            # 确保裁剪区域在图像边界内，与第一个文件的处理逻辑一致
                            top_clamp = max(0, top)
                            left_clamp = max(0, left)
                            bottom_clamp = min(img.shape[0], bottom)
                            right_clamp = min(img.shape[1], right)
                            
                            # 裁剪人脸，与第一个文件的处理逻辑一致
                            face_crop = img[top_clamp:bottom_clamp, left_clamp:right_clamp]
                            if face_crop.size > 0:
                                # 转换为灰度图
                                if len(face_crop.shape) == 3:
                                    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                                cv2.imwrite(output_image_path, face_crop)
                                processed_images += 1
                            else:
                                # 裁剪失败，保存原始图像（转换为灰度图）
                                if len(img.shape) == 3:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                cv2.imwrite(output_image_path, img)
                        else:
                            # 无法读取图像，使用原始方法
                            if crop_face(input_image_path, output_image_path, margin=margin, min_face_size=min_face_size, 
                                       scale_factor=scale_factor, min_neighbors=min_neighbors, use_dnn=use_dnn, tight_crop=tight_crop):
                                processed_images += 1
                    else:
                        # 没有人脸位置信息，使用原始方法
                        if crop_face(input_image_path, output_image_path, margin=margin, min_face_size=min_face_size, 
                                   scale_factor=scale_factor, min_neighbors=min_neighbors, use_dnn=use_dnn, tight_crop=tight_crop):
                            processed_images += 1
    
    print(f"\n处理完成！")
    print(f"总图像数: {total_images}")
    print(f"成功裁剪人脸: {processed_images}")
    print(f"未检测到人脸: {total_images - processed_images}")

if __name__ == "__main__":
    import argparse
    import urllib.request
    import zipfile
    
    # 检查并下载模型文件
    def download_dnn_model():
        # 检查是否需要下载 DNN 模型
        dnn_files = [
            ("deploy.prototxt", "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"),
            ("res10_300x300_ssd_iter_140000_fp16.caffemodel", "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel")
        ]
        
        for filename, url in dnn_files:
            if not os.path.exists(filename):
                print(f"下载 {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"{filename} 下载完成")
                except Exception as e:
                    print(f"下载 {filename} 失败: {str(e)}")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="裁剪数据集中的人脸")
    parser.add_argument('--margin', type=int, default=10, help='人脸边界扩展像素数（默认10）')
    parser.add_argument('--min-face-size', type=int, default=30, help='最小人脸大小（默认30）')
    parser.add_argument('--scale-factor', type=float, default=1.1, help='Haar级联检测器缩放比例（默认1.1）')
    parser.add_argument('--min-neighbors', type=int, default=5, help='Haar级联检测器邻居数（默认5）')
    parser.add_argument('--use-dnn', action='store_true',default=True, help='使用DNN模型进行人脸检测（更精确但需要额外模型文件）')
    parser.add_argument('--tight-crop', action='store_true', default=True, help='进行紧密裁剪（聚焦于面部五官区域并接近正方形）')
    
    args = parser.parse_args()
    
    # 如果启用DNN模式，尝试下载DNN模型
    if args.use_dnn:
        download_dnn_model()
    
    print(f"使用参数: margin={args.margin}, min_face_size={args.min_face_size}, scale_factor={args.scale_factor}, min_neighbors={args.min_neighbors}, use_dnn={args.use_dnn}, tight_crop={args.tight_crop}")
    process_dataset(margin=args.margin, min_face_size=args.min_face_size, 
                   scale_factor=args.scale_factor, min_neighbors=args.min_neighbors, 
                   use_dnn=args.use_dnn, tight_crop=args.tight_crop)