#!/bin/bash

# --- 配置路径 ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
INPUT_DIR="${RAG_OCR_INPUT:-$PROJECT_DIR/data/document.pdf}"
# 注意：如果是单文件，上面的INPUT_DIR直接指向文件；如果是目录请自行调整find逻辑
# 这里假设 INPUT_DIR 是具体文件路径，或者用下面的 find 逻辑扫描整个目录

BASE_INPUT_DIR="${RAG_DATA_DIR:-$PROJECT_DIR/data}" # 用于计算相对路径的根目录
OUTPUT_DIR="${RAG_OCR_OUTPUT:-$PROJECT_DIR/pre_data/product/ocr}"
MODEL_DIR="${RAG_OCR_MODEL_DIR:-$PROJECT_DIR/models/pdocr}"
LAYOUT_DIR="${RAG_LAYOUT_MODEL_DIR:-$PROJECT_DIR/models/PP-DocLayoutV2}"
PYTHON_BIN="${RAG_PYTHON:-python3}"

# 临时目录（存放切割后的 PDF）
TEMP_SPLIT_DIR="${RAG_OCR_TEMP_DIR:-$PROJECT_DIR/pre_data/product/temp_split}"

# 日志路径
LOG_DIR="${RAG_OCR_LOG_DIR:-$PROJECT_DIR/pre_data/product}"
SUCCESS_LOG="${LOG_DIR}/success.log"
FAILED_LOG="${LOG_DIR}/failed.log"
TASK_LOG="${LOG_DIR}/task.log"

# 创建目录
mkdir -p "$OUTPUT_DIR" "$TEMP_SPLIT_DIR" "$LOG_DIR"
touch "$SUCCESS_LOG" "$FAILED_LOG" "$TASK_LOG"

# --- 读取已完成列表 ---
declare -A SKIP_SET
if [[ -f "$SUCCESS_LOG" ]]; then
    while IFS= read -r line; do
        [[ -n "$line" ]] && SKIP_SET["$line"]=1
    done < "$SUCCESS_LOG"
fi

# --- 获取待处理文件 ---
# 如果 INPUT_DIR 直接是文件
if [[ -f "$INPUT_DIR" ]]; then
    FILES=("$INPUT_DIR")
else
    # 如果是目录，则搜索
    mapfile -t FILES < <(find "$INPUT_DIR" -type f -name "*.pdf")
fi

PENDING_FILES=()
for f in "${FILES[@]}"; do
    if [[ -z "${SKIP_SET["$f"]}" ]]; then
        PENDING_FILES+=("$f")
    fi
done

TOTAL=${#PENDING_FILES[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "✅ 没有需要处理的文件" | tee -a "$TASK_LOG"
    exit 0
fi

echo "📌 待处理大文件数: $TOTAL" | tee -a "$TASK_LOG"
echo "🚀 开始执行（双卡并行：GPU 0 & 1）" | tee -a "$TASK_LOG"

# --- 定义处理函数 ---
# 参数：$1=输入PDF小块路径, $2=输出目录, $3=GPU_ID
run_paddle_job() {
    local f_chunk="$1"
    local target_dir="$2"
    local gpu_id="$3"
    
    local tmp_log_file
    tmp_log_file=$(mktemp)

    echo "[GPU $gpu_id] Start: $(basename "$f_chunk")"
    
    # 指定 GPU 并运行
    CUDA_VISIBLE_DEVICES="$gpu_id" paddleocr doc_parser \
        -i "$f_chunk" \
        --save_path "$target_dir" \
        --vl_rec_model_dir "$MODEL_DIR" \
        --layout_detection_model_dir "$LAYOUT_DIR" \
        --device gpu \
        --format_block_content true \
        --use_layout_detection true \
        --use_tensorrt false \
        --precision fp16 >> "$tmp_log_file" 2>&1
    
    local ret_code=$?
    
    # 检查 OOM
    if grep -q "ResourceExhaustedError" "$tmp_log_file"; then
        echo "❌ [GPU $gpu_id] OOM Error: $(basename "$f_chunk")"
        cat "$tmp_log_file" >> "${LOG_DIR}/oom_failed.log"
        rm -f "$tmp_log_file"
        return 1
    elif [[ $ret_code -eq 0 ]]; then
        echo "✅ [GPU $gpu_id] Type: $(basename "$f_chunk")"
        cat "$tmp_log_file" >> "$TASK_LOG"
        rm -f "$tmp_log_file"
        return 0
    else
        echo "❌ [GPU $gpu_id] Failed: $(basename "$f_chunk")"
        cat "$tmp_log_file" >> "$FAILED_LOG"
        rm -f "$tmp_log_file"
        return 1
    fi
}

# --- 主循环 ---
for f in "${PENDING_FILES[@]}"; do
    echo "✂️  正在切割文件: $f" | tee -a "$TASK_LOG"
    
    # 计算相对路径，保持输出目录结构
    rel_path="${f#$INPUT_DIR}"
    # 如果 INPUT_DIR 就是文件，rel_path可能是空的或文件名，做个修正
    if [[ "$f" == "$INPUT_DIR" ]]; then
        file_name=$(basename "$f")
        rel_dir=""
    else
        rel_dir=$(dirname "$rel_path")
    fi
    
    final_target_dir="$OUTPUT_DIR/$rel_dir/$(basename "$f" .pdf)"
    mkdir -p "$final_target_dir"

    # 1. 调用 Python 切割 PDF，每 20 页一片 (避免 OOM)
    # 可通过 RAG_PYTHON 指定本机的 Python 可执行程序
    "$PYTHON_BIN" "$SCRIPT_DIR/split_pdf.py" "$f" "$TEMP_SPLIT_DIR" 1
    
    # 获取切割后的文件列表
    mapfile -t CHUNKS < <(find "$TEMP_SPLIT_DIR" -maxdepth 1 -name "*.pdf" | sort)
    
    # 2. 并行分发给 GPU
    count=0
    for chunk in "${CHUNKS[@]}"; do
        # 简单的轮询调度：偶数去 GPU0，奇数去 GPU1
        gpu_id=$((count % 2))
        
        # 运行任务 (放在后台)
        run_paddle_job "$chunk" "$final_target_dir" "$gpu_id" &
        
        # 记录 PID，用于控制并发数
        pids[${count}]=$!
        
        count=$((count + 1))
        
        # 限制并发：如果已经投递了 2 个任务（占满两张卡），则等待其中一个完成
        # 注意：这里简化逻辑，每投递 2 个任务就全部 wait 一次，确保不会有第 3 个进程抢显存
        if [[ $((count % 2)) -eq 0 ]]; then
            wait
        fi
    done
    
    # 等待剩余任务完成
    wait
    
    # 清理切割的临时文件，准备下一个大文件
    rm -f "$TEMP_SPLIT_DIR"/*.pdf
    
    # 标记大文件处理尝试完成 (无论子块是否全部成功，视为处理过)
    echo "$f" >> "$SUCCESS_LOG"
    echo "🎉 完成处理: $f" | tee -a "$TASK_LOG"

done

echo "🎯 所有任务结束！"
