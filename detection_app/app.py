import gradio as gr
from inference import DeepfakeDetector

def create_interface():
    # 初始化检测器
    detector = DeepfakeDetector(
        '../checkpoints/M2TR_CelebDF_epoch_00019.pyth'
    )
    
    def detect_image(image):
        is_fake, confidence = detector.predict(image)
        # 打印原始预测值进行调试
        print(f"Raw prediction - is_fake: {is_fake}, confidence: {confidence}")
        # 返回字典格式
        return {
            "真实图像": 1 - confidence,
            "伪造图像": confidence
        }
    
    # 创建界面
    interface = gr.Interface(
        fn=detect_image,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=2),
        title="DeepFake人脸检测系统",
        description="上传人脸图片，系统将判断是否为AI生成的伪造图像",
    )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)