# -*- coding: utf-8 -*-
"""从 LCMDC 一级分诊 CSV 抽出 15 科目标 8000 条的文本子集。

原始 CSV 只读。急诊科原标签不存在，只写入数量表，不建文件夹。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_HASH = {
    "label_1_train.csv": "9fd1f5929579ee3c18ddf0aa136752b2e6e9669c1cddef44dd453fdf2e62063e",
    "label_1_test.csv": "c74eadaaea1e7d7061c5656834ab52aad3378551c070aed5e95971834f86f39b",
}
EXPECTED_N = {"train": 373686, "test": 65944}
TARGET_TOTAL = 8000
TEST_EACH = 80
SEED_TEXT = "20261113|{dept}|{split}"
PC1 = ["肿瘤科", "康复医学科", "儿科", "传染科", "辅助检查科", "外科", "中医科"]
PC2 = ["整形美容科", "心理健康科", "妇产科", "皮肤性病科", "内科", "五官科", "营养保健科"]
EMERGENCY = "急诊科"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rng_for(dept: str, split: str) -> random.Random:
    text = SEED_TEXT.format(dept=dept, split=split).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(text).digest()[:8], "big")
    return random.Random(seed)


def load_split(path: Path, split: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="gb18030", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["titles", "infos", "labels_1_word", "labels_1"]:
            raise SystemExit(f"表头不符: {path}")
        for row in reader:
            if not row["infos"].strip() or not row["labels_1_word"].strip():
                raise SystemExit(f"{split} 存在空文本或空科室")
            rows.append(row)
    if len(rows) != EXPECTED_N[split]:
        raise SystemExit(f"{split} 条数 {len(rows)}，期望 {EXPECTED_N[split]}")
    return rows


def sample_indices(n: int, k: int, dept: str, split: str) -> list[int]:
    idxs = list(range(n))
    if n > k:
        idxs = rng_for(dept, split).sample(idxs, k)
    idxs.sort()
    return idxs


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def record_for(split: str, row_no: int, row: dict) -> dict:
    return {
        "id": f"LCMDC-{split}-{row_no:06d}",
        "department": row["labels_1_word"],
        "label_id": int(row["labels_1"]),
        "split": split,
        "title": row["titles"],
        "text": row["infos"],
        "source": "LCMDC",
        "doi": "10.5281/zenodo.13771008",
        "license": "CC BY 4.0",
        "source_file": f"label_1_{split}.csv",
        "source_split": split,
        "source_row": row_no,
    }


def verify(out_dir: Path, raw: dict[str, list[dict]], expected: dict[tuple[str, str], int]) -> None:
    seen_ids = set()
    seen_src = set()
    got = Counter()
    for pc, depts in (("电脑1", PC1), ("电脑2", PC2)):
        root = out_dir / pc
        found = sorted(p.name for p in root.iterdir() if p.is_dir())
        if found != sorted(depts):
            raise SystemExit(f"{pc} 科室目录不符: {found}")
        for dept in depts:
            for split in ("train", "test"):
                path = root / dept / f"{split}.jsonl"
                prev_row = 0
                n = 0
                with path.open(encoding="utf-8", newline="\n") as f:
                    for line in f:
                        obj = json.loads(line)
                        n += 1
                        if obj["id"] in seen_ids:
                            raise SystemExit(f"重复 id {obj['id']}")
                        seen_ids.add(obj["id"])
                        key = (obj["source_split"], obj["source_row"])
                        if key in seen_src or obj["source_split"] != split:
                            raise SystemExit(f"来源位置重复或划分不符 {key}")
                        seen_src.add(key)
                        if obj["source_row"] <= prev_row:
                            raise SystemExit(f"{path} 未按原行号排序")
                        prev_row = obj["source_row"]
                        raw_row = raw[split][obj["source_row"] - 1]
                        if (
                            obj["title"] != raw_row["titles"]
                            or obj["text"] != raw_row["infos"]
                            or obj["department"] != dept
                            or obj["department"] != raw_row["labels_1_word"]
                            or obj["label_id"] != int(raw_row["labels_1"])
                            or obj["split"] != split
                            or obj["id"] != f"LCMDC-{split}-{obj['source_row']:06d}"
                        ):
                            raise SystemExit(f"记录与原 CSV 不一致: {obj['id']}")
                got[(dept, split)] = n
                if n != expected[(dept, split)]:
                    raise SystemExit(f"{dept} {split} 写入 {n}，期望 {expected[(dept, split)]}")
    if (out_dir / EMERGENCY).exists():
        raise SystemExit("急诊科不应有文件夹")


def markdown(rows: list[dict], stats: dict) -> str:
    short = [r for r in rows if r["待补合计"] and r["科室"] != "合计"]
    lines = [
        "# LCMDC 文本抽样划分",
        "",
        "这是给文本模型准备的中文原文抽样。`text` 取原始 `infos`，`title` 取原始 `titles`。没有改写成粤语，也没有生成推荐诊室或原因。",
        "",
        "来源是 LCMDC 一级分诊，许可 CC BY 4.0，DOI 为 [10.5281/zenodo.13771008](https://doi.org/10.5281/zenodo.13771008)。本地保存的 Zenodo 记录许可编号是 `cc-by-4.0`。原始 CSV 不在本仓库，按 `gb18030` 读取。",
        "",
        f"清点结果：训练 {stats['n_train']} 条，测试 {stats['n_test']} 条，合计 {stats['n_total']} 条，14 个科室。空文本 0 条；同一划分内相同 `infos` 0 条；训练和测试之间相同 `infos` 0 条。字段内部换行 {stats['multiline']} 条，行号按 CSV 记录计，不按物理行计。`infos` 最短 {stats['min_len']} 字，中位数 {stats['median_len']} 字，平均 {stats['mean_len']:.1f} 字。本次没有按长度删除。",
        "",
        "原始文件 SHA-256：",
        "",
        f"- `label_1_train.csv`：`{EXPECTED_HASH['label_1_train.csv']}`",
        f"- `label_1_test.csv`：`{EXPECTED_HASH['label_1_test.csv']}`",
        "",
        "## 配额",
        "",
        "目标是训练加测试 8000 条，按 15 个科室计算。现有标签没有急诊科，配额仍给它留位。",
        "",
        "8000 = 15 × 533 + 5。14 个现有科室各 533 条，其中训练 453、测试 80。多出的 5 条放在急诊科训练，所以急诊科是训练 458、测试 80、合计 538。测试没有再拆验证集。",
        "",
        f"每科测试 80 条，是按原始测试占比取整：round(533 × {stats['n_test']} / {stats['n_total']}) = 80。80/533 = 15.01%，原始占比是 {stats['n_test']}/{stats['n_total']} = {stats['test_ratio']:.4%}。",
        "",
        "## 抽样",
        "",
        "只在原训练集或原测试集内部抽，避免训练文本进入测试。每个科室、每个划分单独抽样。种子字符串是 `20261113|{科室}|{train或test}`，取 SHA-256 前 8 字节作为 Python `random.Random` 的整数种子。数量不足的科室全部纳入，不抽样。抽出的记录按原行号升序保存。",
        "",
        "编号格式是 `LCMDC-{原划分}-{原行号六位}`。复现命令：",
        "",
        "```text",
        "python scripts/split_lcmdc_8000.py",
        "```",
        "",
        "原始 CSV 的默认路径在脚本里，可用 `--raw` 改。",
        "",
        "## 两台电脑",
        "",
        "14 个科室按 7+7 分。两个不够的科室分开，其余按原标签编号交错，中医科放在电脑1，使两边都是 7 个科室。急诊科只出现在数量表。",
        "",
        f"- 电脑1：{'、'.join(PC1)}。纳入 {stats['pc1']} 条，目标 {stats['pc_target']} 条，待补 {stats['pc1_short']} 条。",
        f"- 电脑2：{'、'.join(PC2)}。纳入 {stats['pc2']} 条，目标 {stats['pc_target']} 条，待补 {stats['pc2_short']} 条。",
        "",
        "## 待补",
        "",
        "| 科室 | 现有训练 | 现有测试 | 待补训练 | 待补测试 | 待补合计 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in short:
        lines.append(
            f"| {row['科室']} | {row['纳入训练']} | {row['纳入测试']} | {row['待补训练']} | {row['待补测试']} | {row['待补合计']} |"
        )
    lines += [
        "",
        f"其余 12 科原数据足够，各纳入训练 453、测试 80。本次文件共 {stats['written']} 条，待补 {stats['short']} 条。三科补齐前，不能把这套文件当作科室均衡的训练集。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path(r"D:\粤语AI语音导诊\数据集\LCMDC_约44万条\原始数据_约44万条\来源原件\triage\label_1_cls_data"),
    )
    parser.add_argument("--out", type=Path, default=repo / "数据集" / "LCMDC_8000_15科")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.out
    if out.exists():
        if not args.force:
            raise SystemExit(f"输出已存在，加 --force 才会重写: {out}")
        if out.name != "LCMDC_8000_15科":
            raise SystemExit("拒绝删除名称不是 LCMDC_8000_15科 的目录")
        import shutil
        shutil.rmtree(out)

    hashes_before = {name: sha256(args.raw / name) for name in EXPECTED_HASH}
    for name, expect in EXPECTED_HASH.items():
        if hashes_before[name] != expect:
            raise SystemExit(f"{name} 哈希不符，停止")

    raw = {
        "train": load_split(args.raw / "label_1_train.csv", "train"),
        "test": load_split(args.raw / "label_1_test.csv", "test"),
    }
    grouped = {split: defaultdict(list) for split in raw}
    infos = {split: Counter() for split in raw}
    all_infos = {"train": set(), "test": set()}
    multiline = 0
    lengths = []
    for split, rows in raw.items():
        for i, row in enumerate(rows, start=1):
            grouped[split][row["labels_1_word"]].append(i)
            infos[split][row["infos"]] += 1
            all_infos[split].add(row["infos"])
            if "\n" in row["titles"] or "\n" in row["infos"]:
                multiline += 1
            lengths.append(len(row["infos"]))
    for split in raw:
        extra = sum(v - 1 for v in infos[split].values() if v > 1)
        if extra:
            raise SystemExit(f"{split} 有 {extra} 条重复 infos，本脚本不自动去重")
    if all_infos["train"] & all_infos["test"]:
        raise SystemExit("训练和测试存在相同 infos")

    base = TARGET_TOTAL // 15
    remainder = TARGET_TOTAL % 15
    if base != 533 or remainder != 5:
        raise SystemExit("8000 与 15 科的整除结果变了")
    test_each = round(base * EXPECTED_N["test"] / sum(EXPECTED_N.values()))
    if test_each != TEST_EACH:
        raise SystemExit(f"测试配额取整结果是 {test_each}，不是 {TEST_EACH}")
    train_each = base - test_each
    targets = {dept: {"train": train_each, "test": test_each} for dept in PC1 + PC2}
    targets[EMERGENCY] = {"train": train_each + remainder, "test": test_each}
    if sum(v["train"] + v["test"] for v in targets.values()) != TARGET_TOTAL:
        raise SystemExit("配额合计不是 8000")

    pc_of = {dept: "电脑1" for dept in PC1}
    pc_of.update({dept: "电脑2" for dept in PC2})
    expected = {}
    written = 0
    for dept in PC1 + PC2:
        for split in ("train", "test"):
            rows = grouped[split][dept]
            take = min(len(rows), targets[dept][split])
            expected[(dept, split)] = take
            chosen = sample_indices(len(rows), take, dept, split)
            records = [record_for(split, rows[i], raw[split][rows[i] - 1]) for i in chosen]
            write_jsonl(out / pc_of[dept] / dept / f"{split}.jsonl", records)
            written += len(records)

    verify(out, raw, expected)
    hashes_after = {name: sha256(args.raw / name) for name in EXPECTED_HASH}
    if hashes_after != hashes_before:
        raise SystemExit("原始 CSV 哈希发生变化")

    lengths.sort()
    table = []
    order = ["肿瘤科", "整形美容科", "康复医学科", "心理健康科", "儿科", "妇产科", "传染科", "皮肤性病科", "辅助检查科", "内科", "外科", "五官科", "营养保健科", "中医科", EMERGENCY]
    label_ids = {
        "肿瘤科": 0, "整形美容科": 1, "康复医学科": 2, "心理健康科": 3, "儿科": 4,
        "妇产科": 5, "传染科": 6, "皮肤性病科": 7, "辅助检查科": 8, "内科": 9,
        "外科": 10, "五官科": 11, "营养保健科": 12, "中医科": 13, EMERGENCY: "",
    }
    for dept in order:
        orig_train = len(grouped["train"][dept]) if dept != EMERGENCY else 0
        orig_test = len(grouped["test"][dept]) if dept != EMERGENCY else 0
        got_train = expected.get((dept, "train"), 0)
        got_test = expected.get((dept, "test"), 0)
        tgt_train = targets[dept]["train"]
        tgt_test = targets[dept]["test"]
        if dept == EMERGENCY:
            action = "全部待补"
        elif got_train == tgt_train and got_test == tgt_test:
            action = "随机抽样"
        else:
            action = "全部纳入"
        table.append({
            "科室": dept,
            "标签编号": label_ids[dept],
            "电脑": pc_of.get(dept, "后补"),
            "原训练": orig_train,
            "原测试": orig_test,
            "目标训练": tgt_train,
            "目标测试": tgt_test,
            "目标合计": tgt_train + tgt_test,
            "纳入训练": got_train,
            "纳入测试": got_test,
            "纳入合计": got_train + got_test,
            "待补训练": tgt_train - got_train,
            "待补测试": tgt_test - got_test,
            "待补合计": (tgt_train + tgt_test) - (got_train + got_test),
            "处理": action,
        })
    totals = {k: 0 for k in table[0] if k not in ("科室", "标签编号", "电脑", "处理")}
    for row in table:
        for k in totals:
            totals[k] += row[k]
    table.append({
        "科室": "合计", "标签编号": "", "电脑": "", "处理": "", **totals,
    })
    with (out / "诊室数量表.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    pc_target = 7 * base
    stats = {
        "n_train": EXPECTED_N["train"],
        "n_test": EXPECTED_N["test"],
        "n_total": sum(EXPECTED_N.values()),
        "test_ratio": EXPECTED_N["test"] / sum(EXPECTED_N.values()),
        "multiline": multiline,
        "min_len": lengths[0],
        "median_len": lengths[len(lengths) // 2],
        "mean_len": sum(lengths) / len(lengths),
        "written": written,
        "short": totals["待补合计"],
        "pc1": sum(row["纳入合计"] for row in table if row["电脑"] == "电脑1"),
        "pc2": sum(row["纳入合计"] for row in table if row["电脑"] == "电脑2"),
        "pc1_short": sum(row["待补合计"] for row in table if row["电脑"] == "电脑1"),
        "pc2_short": sum(row["待补合计"] for row in table if row["电脑"] == "电脑2"),
        "pc_target": pc_target,
    }
    (out / "划分说明.md").write_text(markdown(table, stats), encoding="utf-8", newline="\n")
    print(json.dumps({k: stats[k] for k in ("written", "short", "pc1", "pc2", "pc1_short", "pc2_short")}, ensure_ascii=False))
    for row in table:
        if row["待补合计"] or row["科室"] in ("合计", EMERGENCY):
            print(f"{row['科室']}\t纳入{row['纳入合计']}\t待补{row['待补合计']}\t{row['处理']}")


if __name__ == "__main__":
    main()