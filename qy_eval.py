import asyncio, aiohttp, json, os, csv, time
from tqdm.asyncio import tqdm_asyncio
import pathlib

# ========== 用户可调参数 ==========
MODELS = [
    # "gpt-4o",
    # "qwen-vl-plus"
    # "Kimi-K2-Instruct"
    ## "gemini-2.5-flash"
    ## "gemini-2.5-pro"
    # "claude-sonnet-4-20250514"
    "gpt-5"
    # "gpt-oss-120b"
    ## "claude-3-7-sonnet-20250219",
    # "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    # "zai-org/GLM-4.5-FP8"
    # "Qwen/Qwen2.5-VL-72B-Instruct"
    # "MiniMax/MiniMax-M1"
    # "gpt-4.1"
    # "google/gemini-2.5-pro"
    # "google/gemini-2.5-flash"
    # "x-ai/grok-4"
]

API_KEY = ""
BASE_URL = ""  # 替换为你的代理地址

SHOTS = 5                       # 0 或 1-5
CONCURRENT = 1                 # 并发请求数
DATA_DIR = r"E:\data\mmlu"          # ← 注意 r 前缀，防止 \ 被转义
DEV_FILE = os.path.join(DATA_DIR, "dev.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
OUT_DIR = pathlib.Path(r"E:\data\mmlu_results")
os.makedirs(OUT_DIR, exist_ok=True)
# ====================================


async def fetch(session, url, headers, payload, sem):
    async with sem:
        async with session.post(url, headers=headers, json=payload) as resp:
            return await resp.json()


def build_prompt(subject, dev_examples, question, choices):
    prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ').title()}.\n\n Answer in uppercase letter only, no extra output."
    for ex in dev_examples:
        prompt += f"{ex['question']}\n"
        for ch, txt in zip(ex["choices"], ex["choice_text"]):
            prompt += f"({ch}) {txt} "
        prompt += f"Answer: {ex['answer']}\n\n"
    prompt += f"{question}\n"
    for ch, txt in choices:
        prompt += f"({ch}) {txt} "
    prompt += "Answer:"
    return prompt


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:   # -sig 去掉 BOM
        return [json.loads(line) for line in f if line.strip()]


async def evaluate_model(model_name):
    dev = read_jsonl(DEV_FILE)
    test = read_jsonl(TEST_FILE)

    dev_by_subj = {}
    for ex in dev:
        dev_by_subj.setdefault(ex["subject"], []).append(ex)

    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{BASE_URL}/chat/completions"

    correct = 0
    results = []

    # # 方案 C：每 BATCH_SIZE 题后休息
    BATCH_SIZE = 20
    SLEEP_BETWEEN_BATCH = 90  # 秒

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(test), BATCH_SIZE):
            batch = test[batch_start:batch_start + BATCH_SIZE]
            tasks = []
            for ex in batch:
                subject = ex["subject"]
                dev_examples = dev_by_subj.get(subject, [])[:SHOTS]
                prompt = build_prompt(
                    subject,
                    dev_examples,
                    ex["question"],
                    list(zip(ex["choices"], ex["choice_text"]))
                )
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "temperature": 1.0,
                    # "stop": ["\n"]
                }
                tasks.append(fetch(session, url, headers, payload, asyncio.Semaphore(1)))

            raw_outputs = await asyncio.gather(*tasks, return_exceptions=True)

            for ex, out in zip(batch, raw_outputs):
                # try:
                if isinstance(out, Exception):
                    pred = None
                    ok = False
                else:
                    # print(out)
                    text = out['choices'][0]["message"]["content"].strip().upper()
                    pred = next((c for c in ex["choices"] if c in text), None)
                    ok = pred == ex["answer"]
                    if ok:
                        correct += 1
                results.append({
                    "model": model_name,
                    "subject": ex["subject"],
                    "question_id": ex["id"],
                    "answer": ex["answer"],
                    "predict": pred,
                    "correct": ok
                })
                # except Exception:
                    # results.append({
                    #     "model": model_name,
                    #     "subject": ex["subject"],
                    #     "question_id": ex["id"],
                    #     "answer": ex["answer"],
                    #     "predict": None,
                    #     "correct": False
                    # })

            print(f"[{model_name}] 完成 {batch_start + len(batch)}/{len(test)}，休息 {SLEEP_BETWEEN_BATCH}s")
            if batch_start + BATCH_SIZE < len(test):
                await asyncio.sleep(SLEEP_BETWEEN_BATCH)

    csv_path = OUT_DIR / f"{model_name.replace('/', '_')}_{SHOTS}shot.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "subject", "question_id", "answer", "predict", "correct"])
        writer.writeheader()
        writer.writerows(results)

    acc = correct / len(test) * 100
    print(f"{model_name}  5-shot accuracy: {acc:.2f}%")
    return acc


async def main():
    tasks = [evaluate_model(m) for m in MODELS]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())