import os, csv, json, glob, pathlib

def csv_dir_to_jsonl(csv_dir, out_path):
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fout:
        csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
        if not csv_files:
            raise FileNotFoundError(f'No CSV files found in {csv_dir}')
        for csv_file in sorted(csv_files):
            subject = os.path.splitext(os.path.basename(csv_file))[0]
            with open(csv_file, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # header
                for idx, row in enumerate(reader):
                    if len(row) != 6:
                        continue
                    q, a, b, c, d, ans = row
                    json.dump({
                        "id": f"{subject}_{idx}",
                        "subject": subject,
                        "question": q.strip(),
                        "choice_text": [a.strip(), b.strip(), c.strip(), d.strip()],
                        "choices": ["A", "B", "C", "D"],
                        "answer": ans.strip().upper()
                    }, fout, ensure_ascii=False)
                    fout.write('\n')
    print(f"已生成 {out_path}  ({sum(1 for _ in open(out_path, encoding='utf-8'))} 条样本)")

if __name__ == '__main__':
    base = pathlib.Path(r"E:\data")
    csv_dir_to_jsonl(base / "dev",  base / "mmlu" / "dev.jsonl")
    csv_dir_to_jsonl(base / "test", base / "mmlu" / "test.jsonl")