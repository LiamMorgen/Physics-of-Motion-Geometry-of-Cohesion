"""
Face Detection Testing Module using InsightFace
This module provides functionality to test if the InsightFace face detection 
models are properly installed and functioning. It attempts to load the face 
detection model, initialize the FaceAnalysis app, and test detection on a 
simple synthetic face image.
The test_face_detector function performs the following steps:
1. Loads the RetinaFace model manually
2. Initializes the FaceAnalysis app with CUDA support
3. Creates a synthetic test image with a simple face
4. Attempts to detect faces in the test image
5. Reports detailed information about detected faces and loaded models
Returns True if all steps succeed, False otherwise.
Dependencies:
- cv2
- numpy
- insightface
Example usage:
    python check_requirements.py
"""
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import insightface.model_zoo as model_zoo
import matplotlib.pyplot as plt

def test_face_detector():
    print("开始测试 insightface 模型...")
    
    try:
        # 尝试手动加载检测模型
        print("尝试加载 retinaface_r50_v1 模型...")
        det_model = model_zoo.get_model('retinaface_r50_v1')
        if det_model is not None:
            print(f"成功加载检测模型: {det_model.taskname}")
        else:
            print("无法加载 retinaface_r50_v1 模型")

        # 初始化 FaceAnalysis
        print("\n初始化 FaceAnalysis...")
        app = FaceAnalysis(providers=['CUDAExecutionProvider'], 
                          allowed_modules=['detection'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("FaceAnalysis 初始化成功")
        
        # 测试一下是否能检测人脸（可选）
        print("\n尝试使用测试图像进行人脸检测...")
        # 创建一个简单的测试图像 (也可以加载真实图像)
        img = np.ones((300, 300, 3), dtype=np.uint8) * 255  # 白色图像
        cv2.circle(img, (150, 120), 70, (0, 0, 0), -1)  # 黑色圆形（模拟头部）
        cv2.circle(img, (125, 100), 10, (255, 255, 255), -1)  # 左眼
        cv2.circle(img, (175, 100), 10, (255, 255, 255), -1)  # 右眼
        cv2.ellipse(img, (150, 150), (30, 15), 0, 0, 180, (255, 255, 255), -1)  # 嘴巴
        
        # 尝试检测
        faces = app.get(img)
        print(f"检测到 {len(faces)} 个人脸")
        
        # 输出详细信息
        if len(faces) > 0:
            print("检测成功！模型工作正常")
            for i, face in enumerate(faces):
                print(f"人脸 {i+1} 位置: {face.bbox}")
                print(f"人脸 {i+1} 关键点: {face.landmark_2d_106.shape if hasattr(face, 'landmark_2d_106') else '无关键点'}")
        
        # 输出应用的模型信息
        print("\n已加载模型信息:")
        for task_name, model in app.models.items():
            print(f"任务: {task_name}, 模型: {model.__class__.__name__}")
        
        return True
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_face_detector()
    print(f"\n测试结果: {'成功' if success else '失败'}")
    
    # 如果测试成功，提供替换建议
    if success:
        print("\n建议修改 landmark_detector.py 文件中的以下代码:")
        print("""
        if model == detectors.RETINAFACE:
            self._face_detector = FaceAnalysis(providers=['CUDAExecutionProvider'])
            self._face_detector.prepare(ctx_id=0, det_size=(224, 224))
        """)