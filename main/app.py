from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import uuid
import traceback
import threading
from typing import Dict, Any, List
import os

# 导入我们在 api.py 中定义的多库生成器
from api import run_rag_session, ENGINES

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

@app.context_processor
def knowledge_base_context():
    definitions = [
        ('employee', '员工培训', 'user', '制度、入职与培训资料', 'local'),
        ('product', '产品信息', 'box', '产品参数与应用资料', 'local'),
        ('secret', '保密信息', 'lock', '因保密原因仅内部网络开放', 'restricted'),
        ('professional', '专业知识库', 'book', '由大模型回答专业问题', 'fallback'),
    ]
    bases = []
    for key, name, icon, description, mode in definitions:
        engine = ENGINES.get(key)
        available = mode == 'fallback' or (engine is not None and not engine.load_error)
        bases.append(dict(id=key, name=name, icon=icon, description=description,
                          available=available, mode=mode,
                          count=len(engine.metadata) if engine and not engine.load_error else 0))
    return dict(knowledge_bases=bases)

# 保存参数
RUNS: Dict[str, Dict[str, Any]] = {}

# 日志缓冲（批量轮询）
LOGS: Dict[str, Dict[str, Any]] = {}
LOGS_LOCK = threading.Lock()

# 定义历史记录和输出文件的目录
HISTORY_FILE = "history.json"
OUTPUT_DIR = "output_history"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return []

def save_history(history: List[Dict[str, Any]]):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def _init_log_channel(run_id: str):
    with LOGS_LOCK:
        if run_id not in LOGS:
            LOGS[run_id] = {"items": [], "done": False}

def _append_log(run_id: str, pretty_line: str, step: Dict[str, Any]):
    try:
        filepath = os.path.join(OUTPUT_DIR, f"{run_id}.txt")
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(pretty_line + '\n')
    except IOError as e:
        print(f"Error: Could not write to log file for run_id {run_id}: {e}")

    with LOGS_LOCK:
        chan = LOGS.get(run_id)
        if not chan:
            chan = {"items": [], "done": False}
            LOGS[run_id] = chan
        seq = len(chan["items"])
        record = {"seq": seq, "pretty": pretty_line, "step": step}
        chan["items"].append(record)

    print(pretty_line, flush=True)

def _mark_done(run_id: str):
    with LOGS_LOCK:
        if run_id in LOGS:
            LOGS[run_id]["done"] = True

def _format_pretty_line(run_id: str, step: Dict[str, Any]) -> str:
    phase = step.get("phase")
    gen_idx = step.get("gen")
    msg = step.get("message", "")
    best_plan = step.get("best_plan")

    line = (
        f"[RUN {run_id[:8]}] phase={phase}"
        + (f", gen={gen_idx}" if gen_idx is not None else "")
        + (f" | {msg}" if msg else "")
    )

    if best_plan:
        plan_str = json.dumps(best_plan, ensure_ascii=False, separators=(',', ':'))
        line += f" | Plan: {plan_str}"

    return line

def start_background_run(run_id: str, params: dict):
    """
    后台线程：消费 RAG 生成器
    """
    try:
        category = params.get("category", "employee")
        head = f"===== 🤖 RAG Session started (run_id={run_id}, KB={category}) ====="
        _append_log(run_id, head, {"phase": "meta", "message": f"started ({category})", "run_id": run_id})

        # 【修改点 1】将 category 传入 API，实现知识库切换
        gen = run_rag_session(
            prompt=params["prompt"],
            category=category
        )

        for step in gen:
            pretty = _format_pretty_line(run_id, step)
            _append_log(run_id, pretty, step)

        tail = f"===== ✅ RAG Session completed (run_id={run_id}) ====="
        _append_log(run_id, tail, {"phase": "final", "message": "completed", "run_id": run_id})
    except Exception as e:
        err = f"===== ❌ RAG Session error (run_id={run_id}): {e} ====="
        _append_log(run_id, err, {"phase": "error", "message": str(e), "run_id": run_id})
        traceback.print_exc()
    finally:
        _mark_done(run_id)

@app.route('/', methods=['GET'])
def index():
    history = load_history()
    return render_template('index.html', history=history)


@app.route('/session/<run_id>', methods=['GET'])
def session_page(run_id):
    prompt = request.args.get('prompt', '')
    history = load_history()
    filepath = os.path.join(OUTPUT_DIR, f"{run_id}.txt")
    
    # --- 新增逻辑开始 ---
    # 1. 尝试从内存(RUNS)获取当前绘画的参数
    current_params = RUNS.get(run_id, {})
    
    # 2. 提取 category。
    # 逻辑：先看内存里有没有 -> 再看URL里有没有 -> 都没有就默认 'employee'
    category = current_params.get("category")
    if not category:
         category = request.args.get('category', 'employee')
    # --- 新增逻辑结束 ---

    # 判断 run_id 是否无效
    is_invalid = (run_id not in RUNS and not os.path.exists(filepath))

    # 【关键修复都在这里】
    # 必须在 render_template 里面加上 category=category
    return render_template(
        'chat.html', 
        run_id=run_id, 
        prompt=prompt, 
        invalid=is_invalid, 
        history=history,
        category=category  # <--- 之前漏掉了这一行！加上它就能正确显示“保密信息”了
    )

@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok"})

@app.route('/solve', methods=['POST'])
def solve_rag():
    try:
        # 取出问题文本
        prompt = request.form.get('query', '').strip()
        if not prompt:
             prompt = request.form.get('vrp_problem', '').strip()

        # 【修改点 2】从前端表单获取 'category'，默认为 'employee'
        # 前端需要在发送 POST 请求时带上 category 字段
        category = request.form.get('category', 'employee')

        params = {
            "prompt": prompt,
            "category": category
        }

        run_id = str(uuid.uuid4())
        RUNS[run_id] = params
        _init_log_channel(run_id)

        # 写入历史
        if prompt:
            history = load_history()
            new_entry = {
                "run_id": run_id, 
                "prompt": prompt,
                "category": category # 记录一下查询类别
            }
            history.insert(0, new_entry)
            save_history(history)

        print("\n" + "=" * 60)
        print("✅ Received /solve (RAG)")
        print(f"  - prompt: {prompt[:120]}..." if prompt else "  - prompt: <EMPTY>")
        print(f"  - category: {category}")
        print(f"➡️  run_id: {run_id}")
        print("=" * 60 + "\n")

        threading.Thread(
            target=start_background_run,
            args=(run_id, dict(params)),
            daemon=True
        ).start()

        return jsonify({"status": "accepted", "run_id": run_id})
    except Exception as e:
        print("❌ /solve error:", e, flush=True)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/logs/<run_id>', methods=['GET'])
def fetch_logs(run_id: str):
    try:
        cursor = int(request.args.get('cursor', 0))
    except Exception:
        cursor = 0

    with LOGS_LOCK:
        chan = LOGS.get(run_id)
        if chan:
            items: List[Dict[str, Any]] = chan["items"]
            done: bool = chan["done"]
            batch = items[cursor:cursor + 200]
            next_cursor = cursor + len(batch)
            payload = [{"seq": rec["seq"], "pretty": rec["pretty"]} for rec in batch]
            return jsonify({
                "status": "ok",
                "logs": payload,
                "next_cursor": next_cursor,
                "done": done
            })

    filepath = os.path.join(OUTPUT_DIR, f"{run_id}.txt")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
            if cursor < len(lines):
                batch_lines = lines[cursor:]
                payload = [{"seq": cursor + i, "pretty": line} for i, line in enumerate(batch_lines)]
                next_cursor = len(lines)
                return jsonify({
                    "status": "ok",
                    "logs": payload,
                    "next_cursor": next_cursor,
                    "done": True
                })
            else:
                return jsonify({
                    "status": "ok",
                    "logs": [],
                    "next_cursor": cursor,
                    "done": True
                })
        except IOError as e:
            return jsonify({"status": "error", "message": f"Could not read log file: {e}"}), 500

    return jsonify({"status": "error", "message": "Invalid run_id"}), 404

if __name__ == '__main__':
    # 端口依然 5000，保持你原有的配置
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
