import os
import argparse
import sys
import copy
import logging
import numpy as np
import warnings

# 1. 显存策略配置
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["FLAGS_eager_delete_tensor_gb"] = "0.0"

# 2. 压制日志和警告
warnings.filterwarnings("ignore")
logging.getLogger('ppocr').setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--rec_model_dir", type=str, required=True)
    parser.add_argument("--layout_model_dir", type=str, required=True)
    parser.add_argument("--rec_batch_num", type=int, default=1)
    parser.add_argument("--det_limit_side_len", type=int, default=960)
    args = parser.parse_args()

    # 导入 PaddleOCR
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("❌ 无法导入 PaddleOCR")
        sys.exit(1)

    print("="*60)
    print("🚀 双显卡流水线 V2 (适配新版接口)")
    print("="*60)

    # =========================================================
    # 步骤 1: 在 GPU 0 上加载 Layout 引擎
    # =========================================================
    print("🔄 [GPU 0] 初始化 Layout 引擎...")
    layout_engine = None
    
    # PaddleOCR V3 不需要手动传 ocr=False, table=False，它默认全加载
    # 我们只要不用它的 OCR 结果即可
    
    # 尝试 A: 带自定义路径
    try:
        layout_engine = PaddleOCR(
            type='structure',
            use_gpu=True,
            gpu_id=0,
            show_log=False,
            # 新版尝试用 image_orientation 做个占位，或者直接忽略路径错误
            layout_model_dir=args.layout_model_dir
        )
        print("✅ Layout 引擎加载成功 (自定义模型)")
    except Exception as e:
        print(f"⚠️ 自定义模型路径参数不被支持 ({e})，切换到默认模型...")
        # 尝试 B: 不带路径 (使用默认)
        try:
            layout_engine = PaddleOCR(
                type='structure',
                use_gpu=True,
                gpu_id=0,
                show_log=False
            )
            print("✅ Layout 引擎加载成功 (官方默认模型)")
        except Exception as e2:
            print(f"❌ Layout 引擎初始化彻底失败: {e2}")
            sys.exit(1)

    # =========================================================
    # 步骤 2: 在 GPU 1 上加载 OCR 引擎
    # =========================================================
    print("🔄 [GPU 1] 初始化 OCR 识别引擎...")
    ocr_engine = None
    try:
        ocr_engine = PaddleOCR(
            use_gpu=True,
            gpu_id=1,
            show_log=False,
            # 同样尝试加载自定义模型，如果失败会自动回退吗？PaddleOCR通常不会
            # 如果这里报错，我们再加 try-except
            rec_model_dir=args.rec_model_dir,
            det_limit_side_len=args.det_limit_side_len 
        )
        print("✅ OCR 引擎加载成功 (GPU 1)")
    except Exception as e:
        print(f"⚠️ OCR自定义模型加载失败 ({e})，尝试使用默认模型...")
        try:
            ocr_engine = PaddleOCR(
                use_gpu=True,
                gpu_id=1,
                show_log=False,
                det_limit_side_len=args.det_limit_side_len
            )
            print("✅ OCR 引擎加载成功 (默认模型)")
        except Exception as e2:
             print(f"❌ OCR 引擎初始化失败: {e2}")
             sys.exit(1)

    # =========================================================
    # 步骤 3: 这里的魔术 —— 手动调度两张卡
    # =========================================================
    if not os.path.exists(args.file_path):
        print(f"❌ 文件不存在: {args.file_path}")
        sys.exit(1)

    save_folder = os.path.join(args.save_path, os.path.basename(args.file_path).split('.')[0])
    os.makedirs(save_folder, exist_ok=True)
    
    print(f"🚀 开始处理: {os.path.basename(args.file_path)}")
    
    try:
        # --- Phase A: GPU 0 进行版面分析 ---
        # 即使 layout_engine 内部跑了 OCR，我们也不管，反正是在 GPU 0 上
        layout_res = layout_engine(args.file_path)
        
        # 兼容性处理
        if isinstance(layout_res, list) and len(layout_res) > 0 and isinstance(layout_res[0], list):
            layout_res = layout_res[0]

        final_results = []
        txt_file_content = []

        print(f"   found {len(layout_res)} regions. sending to GPU 1...")

        # --- Phase B: GPU 1 遍历区域进行识别 ---
        for i, region in enumerate(layout_res):
            region_type = region.get('type', 'unknown')
            
            # 我们只关心图片区域, 重新用 GPU 1 识别文字以保证精度和速度分离
            roi_img = region.get('img')
            
            if isinstance(roi_img, np.ndarray):
                # 无论是 table 还是 text，都扔给 GPU 1 进行文本提取
                # 这样 GPU 0 只是在做切图
                
                # 在 GPU 1 上运行检测+识别
                sub_res = ocr_engine.ocr(roi_img, cls=False, det=True)
                
                text_buffer = ""
                if sub_res and sub_res[0]:
                    for line in sub_res[0]:
                        text_buffer += line[1][0] + "\n"
                
                if text_buffer.strip():
                    txt_file_content.append(f"[{region_type.upper()}]\n{text_buffer}")
                    print(f"     -> Processed {region_type} region {i}")

        # --- Phase C: 保存结果 ---
        txt_path = os.path.join(save_folder, 'result.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(txt_file_content))
            
        print("✅ 处理完成！")

    except Exception as e:
        print(f"❌ 流程中断: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()