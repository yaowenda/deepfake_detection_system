import gradio as gr
from inference import DeepfakeDetector
import time

# 初始化检测器
detector = DeepfakeDetector("../checkpoints/M2TR_CelebDF_epoch_00019.pyth")

def process_media(image, video, progress=gr.Progress()):
    # 选择非空的输入进行处理
    input_media = video if video is not None else image
    if input_media is None:
        return "请上传图片或视频文件", None
    
    # 模拟处理过程并输出日志
    progress(0, desc="初始化检测环境...")
    time.sleep(0.5)
    
    progress(0.2, desc="加载媒体文件...")
    time.sleep(0.5)
    
    progress(0.4, desc="预处理数据...")
    time.sleep(0.5)
    
    progress(0.6, desc="运行深度伪造检测...")
    is_fake, fake_prob, processed_image = detector.predict(input_media) 
    
    progress(0.8, desc="分析检测结果...")
    time.sleep(0.5)
    
    progress(1.0, desc="完成检测!")
    time.sleep(0.3)
    
    # 生成详细的检测报告
    result = "伪造" if is_fake else "真实"
    confidence = fake_prob if is_fake else (1 - fake_prob)
    
    report = f"""
    🔍 深度伪造检测报告
    ═══════════════════════
    
    📊 检测结果: {result}
    🎯 置信度: {confidence:.2%}
    
    🔒 安全评估:
    {'⚠️ 警告：该媒体文件可能是AI生成的伪造内容！' if is_fake else '✅ 该媒体文件未发现伪造痕迹。'}
    
    📝 技术细节:
    - 模型: M2TR DeepFake Detector
    - 检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    return report, processed_image 

def create_interface():
    # 自定义CSS样式
    custom_css = """
    .gradio-container {
        background: linear-gradient(to bottom right, #1a1a2e, #16213e) !important;
    }
    .gradio-interface {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
    }
    """
    
    # 创建界面
    interface = gr.Interface(
        fn=process_media,
        inputs=[
            gr.Image(type="filepath", label="📸 上传图片", elem_classes="input-image"),
            gr.Video(label="🎥 上传视频", elem_classes="input-video")
        ],
        outputs=[
            gr.Textbox(label="📋 检测报告", elem_classes="output-report"),
            gr.Image(label="🖼️ 检测结果可视化", elem_classes="output-image")
        ],
        title="深度伪造检测平台",
        description="""
        ### 本平台基于多模态多尺度Transformer模型，使用Gradio技术搭建

        *注：图片检测需要几秒钟时间，视频检测时间略长，请耐心等待*
        """,
        theme="dark",
        css=custom_css,
        allow_flagging="never"
    )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)