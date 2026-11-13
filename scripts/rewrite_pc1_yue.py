# -*- coding: utf-8 -*-
"""电脑1七科：沿用已有划分，改写成粤语单轮输入输出。不重抽 LCMDC。"""
from __future__ import annotations

import argparse
import csv
import html
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import opencc

PC1 = ["肿瘤科", "康复医学科", "儿科", "传染科", "辅助检查科", "外科", "中医科"]
DEPT_YUE = {
    "肿瘤科": "腫瘤科",
    "康复医学科": "康復醫學科",
    "儿科": "兒科",
    "传染科": "傳染科",
    "辅助检查科": "輔助檢查科",
    "外科": "外科",
    "中医科": "中醫科",
}
TRAIN_POOL_TARGET = 453
TEST_TARGET = 80
RAW_DEFAULT = Path(r"D:\粤语AI语音导诊\数据集\LCMDC_约44万条\原始数据_约44万条\来源原件\triage\label_1_cls_data")
WEB_DEFAULT_NAME = Path("数据集") / "webMedQA_两科候选_清洗前" / "辅助检查科" / "questions.jsonl"
CSV_HASH = {
    "label_1_train.csv": "9fd1f5929579ee3c18ddf0aa136752b2e6e9669c1cddef44dd453fdf2e62063e",
    "label_1_test.csv": "c74eadaaea1e7d7061c5656834ab52aad3378551c070aed5e95971834f86f39b",
}

# 人工看过的边界样本。规则结果不一致时以这份为准。
FORCE_DROP = {
    "321772": "已看过：问的是谁发现X线，不是检查诉求",
    "294466": "已看过：问报告单该怎么写，是工作人员写法",
    "259999": "已看过：问CT片质量控制，不是患者检查诉求",
    "30387799": "已看过：声带增厚、声音沙哑，不是辅助检查",
    "29395903": "已看过：问身材和胯骨，不是辅助检查；与29395907同文",
    "29395907": "已看过：问身材和胯骨，不是辅助检查；与29395903同文",
    "1285590": "已看过：只问阳性之后怎么治疗，不是在问检查结果",
}
FORCE_KEEP = {
    "718679": "已看过：问尿汞检查哪里做、尿检能不能测",
    "804637": "已看过：问精液和前列腺要做什么检查、费用",
    "1583392": "已看过：问痛风要做哪些检查才能确诊",
    "2479570": "已看过：贴了尿检结果，问是否正常",
    "9697097": "已看过：问微量元素和粪便检查前怎么准备、费用",
    "11170701": "已看过：问阿莫西林会不会影响内分泌化验",
    "11201264": "已看过：问睡眠和鼻塞会不会影响内分泌化验",
    "11657566": "已看过：问CCP结果的意思，并附有数值",
    "13672101": "已看过：内容是生化全项检查结果",
    "15281471": "已看过：贴了精子检验数值再提问",
    "29320329": "已看过：尿检后尿蛋白加号，问严不严重",
    "35505209": "已看过：问甲胎蛋白和癌胚抗原数值代表什么",
}

EXAM = [
    "核磁", "磁共振", "磁力共振", "MRI", "MRA", "CT", "B超", "彩超", "超声", "X线", "X光", "胸片", "平片",
    "造影", "化验", "验血", "验肝", "验尿", "血常规", "尿常规", "尿培养", "中段尿", "尿检", "血检",
    "病理", "活检", "脑脊液", "肠镜", "胃镜", "穿刺", "钼靶", "骨密度", "心电图", "脑电图", "肌电图",
    "肺功能", "造影剂", "药敏", "培养", "参考值", "参考范围", "化验单", "检查单", "报告单", "检查报告", "化验报告",
    "CEA", "AFP", "TCT", "HPV", "甲功", "转氨酶", "糖化", "血沉", "血小板", "白细胞", "红细胞",
    "血红蛋白", "隐血", "镜检", "PET", "DSA", "同位素", "平扫", "增强扫描", "影像报告",
    "抽血", "大便常规", "便常规", "粪常规", "粪检", "便检", "骨扫描", "宫腔镜", "支气管镜", "喉镜", "基因检测",
    "染色体", "唐氏", "唐筛", "糖耐", "胰岛素", "乙肝两对半", "乙肝五项", "丙肝", "HIV", "梅毒",
    "凝血", "肝功能", "肾功能", "肝功", "肾功", "血脂", "尿酸", "甲状腺功能", "激素六项", "性激素", "激素五项", "激素5项",
    "肿瘤标志", "CA125", "CA199", "PSA", "钡餐", "钡灌", "动态心电图", "核医学", "放射检查", "剂量",
    "毫西弗", "精液", "精子", "前列腺液", "微量元素", "内分泌检查", "甲胎", "癌胚", "尿蛋白", "血糖",
    "生化", "四维", "胸透", "拍片", "DR", "呼气", "碳13", "碳14", "C13", "C14", "幽门",
    "HCG", "孕酮", "睾酮", "皮质醇", "促肾上腺", "FSH", "LH", "泌乳素", "同型半胱氨酸", "抗O",
    "类风湿因子", "抗核", "ENA", "CCP", "RF", "ASO", "RPR", "TPPA", "支原体", "衣原体", "白带",
    "阴道镜", "无创", "羊水", "胎心监护", "胎监", "脐血流", "阴超", "心脏彩超",
    "骨龄", "TCD", "多普勒", "过敏原", "C肽", "糖化血红蛋白", "乙肝DNA", "HBV", "HCV", "淋球",
    "精子碎片", "尿流", "膀胱镜", "肾图", "静脉肾盂", "腹平片", "乳腺钼靶", "全景片", "牙片",
    "听力检查", "声导抗", "耳声发射", "眼底", "视野", "尿汞", "血汞", "血型", "空腹", "禁食", "憋尿",
    "肠道准备", "辐射", "化验结果", "检查结果", "检验", "标本", "涂片",
    "SSA", "SSB", "抗SSA", "抗SSB",
]
WHICH_TEST = re.compile(
    r"(做哪些检查|哪些检查|做什么检查|做什么化验|做哪些化验|检查些什么|该检测些什么|"
    r"怎样的检查|什么样的检查|检查什么|该检查什么|要检查什么|需做.{0,6}检查|要做.{0,8}检查|"
    r"做什么项目|检查项目|须检查|须做哪些|做什么检查)"
)
MEANING = re.compile(
    r"(什么意思|啥意思|是什么意思|正常吗|是否正常|算不算|严重吗|怎么看|如何看|代表什么|说明什么|"
    r"临床意义|怎么回事|有没有事|要紧|高不高|低不低|偏高|偏低|有问题吗|有没有问题)"
)
PREP = re.compile(
    r"(多少钱|费用|价位|贵吗|空腹|禁食|准备|辐射|副作用|影响结果|影响检查|影响.{0,6}验|"
    r"影响.{0,8}检查|复查|重做|准确|误诊|哪里做|哪家医院|哪个医院|能做吗|可以做吗|"
    r"多久|多长时间|疼不疼|痛不痛|有没有辐射|有害|危害|注意事项)"
)
TREAT = re.compile(r"(怎么治|如何治疗|怎样治疗|怎么治疗|吃什么药|用什么药|吃啥药|如何用药|什么药|怎样康复|怎么康复|如何好)")
TRIVIA = re.compile(r"(第一个发现|是谁发明|是谁发现|质量控制|质量控制因数)")
PROFESSIONAL = re.compile(r"(报告单中应该怎么写|应该怎么写|早班|中班|晚班|技师|上班时间|招人|从业)")
POPSCI = re.compile(r"(为什么叫|有什么局限性|什么叫平片)")
LAB_NUM = re.compile(r"(\d+\.\d+|\d+\s*(mg|mmol|mol|U/L|g/L|%|次/分)|[A-Za-z]{1,8}\s*[:：=]?\s*\d)")
RESULT_WORD = re.compile(r"(结果|偏高|偏低|正常|异常|阳性|阴性|抗体|算不算|参考|报告|指标|增高|降低|箭头|↑|↓)")

# 先长后短。只改高频书面词，不改病名。单字替换都有保护项。
PHRASES = [
    ("问题补充", ""),
    ("磁共振", "磁力共振"),
    ("核磁共振", "磁力共振"),
    ("是不是", "係唔係"),
    ("是不是", "係唔係"),
    ("能不能", "可唔可以"),
    ("可不可以", "可唔可以"),
    ("有没有", "有冇"),
    ("为什么", "點解"),
    ("为啥", "點解"),
    ("怎么回事", "咩事"),
    ("怎么样", "點樣"),
    ("怎样", "點樣"),
    ("怎么办", "點算"),
    ("多少钱", "幾多錢"),
    ("多少", "幾多"),
    ("多久", "幾耐"),
    ("哪里", "邊度"),
    ("哪儿", "邊度"),
    ("哪个", "邊個"),
    ("哪些", "邊啲"),
    ("哪种", "邊種"),
    ("什么", "咩"),
    ("如何", "點樣"),
    ("怎么", "點"),
    ("是否", "係咪"),
    ("不是", "唔係"),
    ("不会", "唔會"),
    ("不能", "唔可以"),
    ("不要", "唔好"),
    ("不用", "唔使"),
    ("不知道", "唔知"),
    ("不清楚", "唔清楚"),
    ("不行", "唔得"),
    ("不好", "唔好"),
    ("不够", "唔夠"),
    ("不太", "唔太"),
    ("不舒服", "唔舒服"),
    ("不想", "唔想"),
    ("不敢", "唔敢"),
    ("不必", "唔使"),
    ("不同", "唔同"),
    ("不过", "不過"),
    ("没有", "冇"),
    ("没事", "冇事"),
    ("没关系", "冇所謂"),
    ("没", "冇"),
    ("其他", "其他"),
    ("他们", "佢哋"),
    ("她们", "佢哋"),
    ("它们", "佢哋"),
    ("我们", "我哋"),
    ("你们", "你哋"),
    ("咱们", "我哋"),
    ("这个", "呢個"),
    ("那个", "嗰個"),
    ("这些", "呢啲"),
    ("那些", "嗰啲"),
    ("这种", "呢種"),
    ("那种", "嗰種"),
    ("这里", "呢度"),
    ("那里", "嗰度"),
    ("这样", "咁樣"),
    ("那样", "咁樣"),
    ("这么", "咁"),
    ("那么", "咁"),
    ("现在", "而家"),
    ("正在", "喺度"),
    ("今天", "今日"),
    ("昨天", "尋日"),
    ("明天", "聽日"),
    ("前天", "前日"),
    ("后天", "後日"),
    ("每天", "每日"),
    ("老是", "成日"),
    ("总是", "成日"),
    ("经常", "成日"),
    ("常常", "成日"),
    ("有时候", "有時"),
    ("有时", "有時"),
    ("偶尔", "間中"),
    ("已经", "已經"),
    ("以后", "之後"),
    ("之后", "之後"),
    ("后来", "之後"),
    ("然后", "跟住"),
    ("接着", "跟住"),
    ("但是", "但係"),
    ("可是", "但係"),
    ("还是", "定係"),
    ("或是", "或者"),
    ("就是", "就係"),
    ("只是", "淨係"),
    ("只有", "淨係"),
    ("要是", "如果"),
    ("因为", "因為"),
    ("所以", "所以"),
    ("如果", "如果"),
    ("应该", "應該"),
    ("可能", "可能"),
    ("需要", "需要"),
    ("希望", "希望"),
    ("想要", "想"),
    ("可以", "可以"),
    ("一下", "一下"),
    ("一会儿", "一陣"),
    ("一会", "一陣"),
    ("有点", "有啲"),
    ("有些", "有啲"),
    ("一点", "一啲"),
    ("一些", "一啲"),
    ("很多", "好多"),
    ("很少", "好少"),
    ("很难", "好難"),
    ("很", "好"),
    ("特别", "特別"),
    ("比较", "比較"),
    ("孩子", "小朋友"),
    ("小孩", "小朋友"),
    ("宝宝", "BB"),
    ("男孩", "男仔"),
    ("女孩", "女仔"),
    ("爸爸", "爸爸"),
    ("妈妈", "媽媽"),
    ("父亲", "爸爸"),
    ("母亲", "媽媽"),
    ("丈夫", "老公"),
    ("妻子", "老婆"),
    ("睡觉", "瞓覺"),
    ("想睡", "想瞓"),
    ("吃饭", "食飯"),
    ("吃药", "食藥"),
    ("吃药", "食藥"),
    ("喝水", "飲水"),
    ("看见", "睇到"),
    ("看到", "睇到"),
    ("看看", "睇下"),
    ("看一看", "睇下"),
    ("看病", "睇病"),
    ("看医生", "睇醫生"),
    ("看起来", "睇落"),
    ("看来", "睇嚟"),
    ("说话", "講嘢"),
    ("不要说", "唔好話"),
    ("医生说", "醫生話"),
    ("说是", "話係"),
    ("做了", "做咗"),
    ("吃了", "食咗"),
    ("喝了", "飲咗"),
    ("去了", "去咗"),
    ("来了", "嚟咗"),
    ("看了", "睇咗"),
    ("查了", "查咗"),
    ("拍了", "照咗"),
    ("拍的", "照咗"),
    ("好了", "好咗"),
    ("得了", "得咗"),
    ("有了", "有咗"),
    ("用了", "用咗"),
    ("长了", "長咗"),
    ("掉了", "甩咗"),
    ("吃过", "食過"),
    ("做过", "做過"),
    ("看过", "睇過"),
    ("去过", "去過"),
    ("了吗", "咗未"),
    ("头疼", "頭痛"),
    ("头痛", "頭痛"),
    ("肚子疼", "肚痛"),
    ("肚子痛", "肚痛"),
    ("拉肚子", "肚屙"),
    ("腹泻", "肚屙"),
    ("想吐", "想嘔"),
    ("恶心", "想嘔"),
    ("呕吐", "嘔"),
    ("咳嗽", "咳"),
    ("发烧", "發燒"),
    ("尿血", "屙血尿"),
    ("便血", "屙血"),
    ("没知觉", "冇知覺"),
    ("没有知觉", "冇知覺"),
    ("刀口", "傷口"),
    ("好转", "好返"),
    ("不见好", "都仲未好"),
    ("复查", "覆查"),
    ("拍片", "照片"),
    ("吸烟", "食煙"),
    ("抽烟", "食煙"),
    ("掉头发", "甩頭髮"),
    ("脱发", "甩髮"),
    ("头发", "頭髮"),
    ("贵不贵", "貴唔貴"),
    ("好不好", "好唔好"),
    ("会不会", "會唔會"),
    ("能不能", "可唔可以"),
    ("给我", "畀我"),
    ("给你", "畀你"),
    ("给他", "畀佢"),
    ("给她", "畀佢"),
    ("在家", "喺屋企"),
    ("在医院", "喺醫院"),
    ("存在", "存在"),
    ("潜在", "潛在"),
    ("还原", "還原"),
    ("归还", "歸還"),
    ("饱和", "飽和"),
    ("温和", "溫和"),
    ("平和", "平和"),
    ("跟踪", "跟蹤"),
    ("跟腱", "跟腱"),
    ("说明", "說明"),
    ("小说", "小說"),
    ("目的", "目的"),
    ("似的", "咁"),
    ("但是", "但係"),
    ("他", "佢"),
    ("她", "佢"),
    ("是", "係"),
    ("在", "喺"),
    ("的", "嘅"),
    ("看", "睇"),
    ("说", "話"),
    ("吃", "食"),
    ("喝", "飲"),
    ("给", "畀"),
    ("很", "好"),
    ("还", "仲"),
    ("吗", "呀"),
    ("啥", "咩"),
    ("吧", "啦"),
    ("啊", "呀"),
    ("疼", "痛"),
    ("和", "同"),
    ("与", "同"),
    ("跟", "同"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_key(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def combined_text(title: str, info: str) -> str:
    title = (title or "").strip()
    info = (info or "").strip()
    if info.startswith(title) or not title:
        return info
    if not info:
        return title
    return title + "\n" + info


def patient_text(title: str, info: str) -> str:
    """给模型看的正文。标题只是截断前缀时不再重复拼上。"""
    title = (title or "").strip()
    info = (info or "").strip()
    if not info:
        return title
    short = re.sub(r"[.。…]+$", "", title).strip()
    if not title or info.startswith(title) or (short and info.startswith(short)):
        return info
    return title + "\n" + info


def case_id_of(text: str) -> str:
    return hashlib.sha256(text_key(text).encode("utf-8")).hexdigest()


def lcmdc_validation(case_id: str) -> bool:
    return int(case_id[:8], 16) % 10 == 0


def clean_question(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    text = text.replace("BR>", "\n").replace("<br>", "\n").replace("<BR>", "\n")
    text = re.sub(r"第[一二三四五六七八九十0-9]+次问题补充[:：]?", "\n", text)
    text = re.sub(r"问题补充[:：]?", "\n", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def has_exam(text: str) -> bool:
    low = text.lower()
    for key in EXAM:
        if key.lower() in low:
            return True
    if WHICH_TEST.search(text) or re.search(r"检查要做|检查方法", text):
        return True
    if re.search(r"回声|透声|正位片|侧位片|放射科|密度影|波幅|测血压|CDFI|血流信号", text):
        return True
    if re.search(r"BPD\s*\d|(?<![A-Za-z])(?:HC|AC|FL)\s*\d", text, re.I):
        return True
    if LAB_NUM.search(text) and RESULT_WORD.search(text):
        return True
    if re.search(r"乙肝|指标", text) and RESULT_WORD.search(text):
        return True
    return False


def norm_dup(text: str) -> str:
    return re.sub(r"\s+", "", clean_question(text))


def decide_web(row: dict, seen: dict) -> tuple[str, str]:
    qid = str(row["question_id"])
    question = clean_question(row["question"])
    key = norm_dup(question)
    if key in seen:
        rule_decision, rule_reason = "删除", "和更早的一条问句完全相同"
    else:
        seen[key] = qid
        exam = has_exam(question)
        numeric = bool(re.search(r"\d", question))
        if TRIVIA.search(question) or PROFESSIONAL.search(question):
            rule_decision, rule_reason = "删除", "科普、历史或工作人员问题"
        elif POPSCI.search(question) and not MEANING.search(question):
            rule_decision, rule_reason = "删除", "问的是技术名词，不是患者结果"
        elif row["candidate_role"] == "widened":
            prep = bool(PREP.search(question))
            result = bool(RESULT_WORD.search(question))
            if exam and (numeric or result or prep):
                rule_decision, rule_reason = "保留", "体检类里有具体结果、数值或检查准备"
            else:
                rule_decision, rule_reason = "删除", "体检类没有具体结果、数值或检查准备"
        elif not exam:
            rule_decision, rule_reason = "删除", "看不出检查、化验或结果"
        elif TREAT.search(question) and not MEANING.search(question):
            rule_decision, rule_reason = "删除", "主要在问治疗或用药，不是在问结果"
        else:
            rule_decision, rule_reason = "保留", "在问检查、化验、结果或检查准备"
    if qid in FORCE_DROP:
        return "删除", FORCE_DROP[qid]
    if qid in FORCE_KEEP:
        return "保留", FORCE_KEEP[qid]
    return rule_decision, rule_reason


class Rewriter:
    def __init__(self) -> None:
        self.cc = opencc.OpenCC("s2hk")
        items = []
        seen = set()
        for src, dst in PHRASES:
            if src in seen:
                continue
            seen.add(src)
            items.append((src, dst))
        items.sort(key=lambda kv: len(kv[0]), reverse=True)
        self.mapping = {src: dst for src, dst in items}
        self.pattern = re.compile("|".join(re.escape(src) for src, _ in items))
        self.day_pattern = re.compile(r"(?<![今昨明前后當当])(\d+)\s*天")

    def to_yue(self, text: str) -> str:
        text = clean_question(text)
        text = self.day_pattern.sub(r"\1日", text)
        # “还是不/没”在主诉里通常是“仍然”，不是选择问句。
        text = re.sub(r"还是(?=不|没)", "仍然", text)
        text = self.pattern.sub(lambda m: self.mapping[m.group(0)], text)
        text = self.cc.convert(text)
        text = text.replace("?", "？").replace("!", "！").replace(",", "，")
        text = re.sub(r"[ ]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"，{2,}", "，", text)
        return text.strip()


def first_piece(text: str) -> str:
    parts = re.split(r"[。！？\n]", text)
    for part in parts:
        part = part.strip()
        if part:
            return part
    return text.strip()


def reason_for(rewriter: Rewriter, title: str, text: str, dept_yue: str) -> str:
    source = first_piece(patient_text(title, text))
    body = rewriter.to_yue(source)
    body = re.sub(r"[？?。！!]+$", "", body).strip("，,、 ")
    body = re.sub(r"\s+", "", body)
    if len(body) > 36:
        cut = body[:36]
        for mark in ("，", "、", "；"):
            idx = cut.rfind(mark)
            if idx >= 12:
                cut = cut[:idx]
                break
        else:
            cut = cut + "…"
        body = cut
    if not body:
        body = "呢則就診提問"
    return f"{body}，適合由{dept_yue}評估。"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_csv(path: Path) -> list[dict]:
    csv.field_size_limit(10**7)
    with path.open(encoding="gb18030", newline="") as f:
        return list(csv.DictReader(f))


def build(repo: Path, raw: Path, write: bool) -> dict:
    for name, expect in CSV_HASH.items():
        got = sha256_file(raw / name)
        if got != expect:
            raise SystemExit(f"{name} 哈希不符，停止")
    raw_rows = {
        "train": load_csv(raw / "label_1_train.csv"),
        "test": load_csv(raw / "label_1_test.csv"),
    }
    before = {}
    root = repo / "数据集" / "LCMDC_8000_15科" / "电脑1"
    for dept in PC1:
        for split in ("train", "test"):
            path = root / dept / f"{split}.jsonl"
            before[str(path)] = sha256_file(path)

    rewriter = Rewriter()
    web_rows = read_jsonl(repo / WEB_DEFAULT_NAME)
    if len(web_rows) != 482:
        raise SystemExit(f"辅助检查科候选不是 482 条，而是 {len(web_rows)}")
    seen = {}
    decisions = []
    kept_web = []
    for row in web_rows:
        decision, reason = decide_web(row, seen)
        split = {"train": "train", "valid": "validation", "test": "test"}[row["source_split"]]
        use = decision == "保留"
        decisions.append({
            "question_id": str(row["question_id"]),
            "source_split": row["source_split"],
            "candidate_role": row["candidate_role"],
            "source_category": row["source_category"],
            "决定": decision,
            "原因": reason,
            "并入": use,
            "并入划分": split if use else None,
        })
        if use:
            kept_web.append((split, row))
    for qid, expect_drop in FORCE_DROP.items():
        hit = [d for d in decisions if d["question_id"] == qid]
        if len(hit) != 1 or hit[0]["决定"] != "删除":
            raise SystemExit(f"删除名单没有生效: {qid}")
    for qid in FORCE_KEEP:
        hit = [d for d in decisions if d["question_id"] == qid]
        if len(hit) != 1 or hit[0]["决定"] != "保留":
            raise SystemExit(f"保留名单没有生效: {qid}")

    summary = {}
    outputs = {}
    occupied = defaultdict(set)
    for dept in PC1:
        for split in ("train", "test"):
            for obj in read_jsonl(root / dept / f"{split}.jsonl"):
                full = combined_text(obj["title"], obj["text"])
                cid = case_id_of(full)
                derived = "test" if split == "test" else ("validation" if lcmdc_validation(cid) else "train")
                occupied[text_key(clean_question(patient_text(obj["title"], obj["text"])))].add((dept, derived))
    for dept in PC1:
        buckets = {"train": [], "validation": [], "test": []}
        metas = {"train": [], "validation": [], "test": []}
        added = Counter()
        for split in ("train", "test"):
            for obj in read_jsonl(root / dept / f"{split}.jsonl"):
                src = raw_rows[split][obj["source_row"] - 1]
                if obj["title"] != src["titles"] or obj["text"] != src["infos"] or obj["department"] != src["labels_1_word"]:
                    raise SystemExit(f"和原始CSV不一致: {obj['id']}")
                full = combined_text(obj["title"], obj["text"])
                shown = patient_text(obj["title"], obj["text"])
                cid = case_id_of(full)
                if split == "test":
                    derived = "test"
                    rule = "LCMDC原测试集"
                else:
                    derived = "validation" if lcmdc_validation(cid) else "train"
                    rule = "LCMDC训练集按case_id前8位十六进制模10等于0进验证集"
                dept_yue = DEPT_YUE[dept]
                record = {
                    "input": rewriter.to_yue(shown),
                    "output": {
                        "建議科室": [dept_yue],
                        "簡單原因": reason_for(rewriter, obj["title"], obj["text"], dept_yue),
                    },
                }
                meta = {
                    "id": obj["id"],
                    "case_id": cid,
                    "source": "LCMDC",
                    "source_split": split,
                    "derived_split": derived,
                    "split_rule": rule,
                    "source_row": obj["source_row"],
                    "department_zh": dept,
                    "license": obj["license"],
                    "doi": obj["doi"],
                    "rewrite": "规则替换加OpenCC_s2hk，不是人工翻译，也没有按CCPC逐句对齐",
                }
                buckets[derived].append(record)
                metas[derived].append(meta)
                added[("lcmdc", derived)] += 1
        if dept == "辅助检查科":
            for derived, row in kept_web:
                full = clean_question(row["question"])
                norm = text_key(full)
                if norm in occupied:
                    where = "、".join(f"{a}{b}" for a, b in sorted(occupied[norm]))
                    for item in decisions:
                        if item["question_id"] == str(row["question_id"]):
                            item["并入"] = False
                            item["并入划分"] = None
                            item["原因"] = item["原因"] + f"；与已纳入文本相同（{where}），不并入，避免串集"
                    continue
                occupied[norm].add((dept, derived))
                dept_yue = DEPT_YUE[dept]
                title = full.split("\n", 1)[0]
                record = {
                    "input": rewriter.to_yue(full),
                    "output": {
                        "建議科室": [dept_yue],
                        "簡單原因": reason_for(rewriter, title, full, dept_yue),
                    },
                }
                meta = {
                    "id": row["id"],
                    "case_id": case_id_of(full),
                    "source": "webMedQA",
                    "source_split": row["source_split"],
                    "derived_split": derived,
                    "split_rule": "webMedQA原划分，valid进入验证集，不并入LCMDC测试集，也不并入训练",
                    "question_id": str(row["question_id"]),
                    "candidate_role": row["candidate_role"],
                    "department_zh": dept,
                    "license": "Apache-2.0",
                    "doi": "10.1186/s12911-019-0761-8",
                    "rewrite": "规则替换加OpenCC_s2hk，不是人工翻译，也没有按CCPC逐句对齐",
                }
                buckets[derived].append(record)
                metas[derived].append(meta)
                added[("web", derived)] += 1
        for derived in buckets:
            ids = [m["id"] for m in metas[derived]]
            if len(ids) != len(set(ids)):
                raise SystemExit(f"{dept} {derived} 有重复id")
            for rec, meta in zip(buckets[derived], metas[derived]):
                if not rec["input"].strip():
                    raise SystemExit(f"空输入 {meta['id']}")
                if rec["output"]["建議科室"] != [DEPT_YUE[dept]]:
                    raise SystemExit(f"科室不一致 {meta['id']}")
                if "適合由" not in rec["output"]["簡單原因"]:
                    raise SystemExit(f"原因缺少分流句 {meta['id']}")
        pool = added[("lcmdc", "train")] + added[("lcmdc", "validation")] + added[("web", "train")]
        test_n = added[("lcmdc", "test")] + added[("web", "test")]
        summary[dept] = {
            "train": len(buckets["train"]),
            "validation": len(buckets["validation"]),
            "test": len(buckets["test"]),
            "lcmdc_train_pool": added[("lcmdc", "train")] + added[("lcmdc", "validation")],
            "lcmdc_validation": added[("lcmdc", "validation")],
            "web_train": added[("web", "train")],
            "web_validation": added[("web", "validation")],
            "web_test": added[("web", "test")],
            "train_pool_short": TRAIN_POOL_TARGET - pool,
            "test_short": TEST_TARGET - test_n,
        }
        outputs[dept] = (buckets, metas)

    seen_input = {}
    for dept, (buckets, _metas) in outputs.items():
        for derived, rows in buckets.items():
            for rec in rows:
                key = text_key(rec["input"])
                prev = seen_input.get(key)
                if prev and prev[1] != derived:
                    raise SystemExit(f"同一正文进入了两个划分: {prev} vs {(dept, derived)}")
                seen_input.setdefault(key, (dept, derived))

    if write:
        for dept, (buckets, metas) in outputs.items():
            out_dir = root / dept / "粤语"
            for derived in ("train", "validation", "test"):
                write_jsonl(out_dir / f"{derived}.jsonl", buckets[derived])
                write_jsonl(out_dir / f"{derived}.meta.jsonl", metas[derived])
        write_jsonl(root / "辅助检查科" / "webmedqa_清洗决定.jsonl", decisions)
        lines = ["科室,训练,验证,测试,合计,LCMDC训练池,其中验证,补入训练,补入验证,补入测试,训练池仍缺,测试仍缺"]
        for dept in PC1:
            item = summary[dept]
            total = item["train"] + item["validation"] + item["test"]
            lines.append(
                f"{dept},{item['train']},{item['validation']},{item['test']},{total},"
                f"{item['lcmdc_train_pool']},{item['lcmdc_validation']},{item['web_train']},"
                f"{item['web_validation']},{item['web_test']},{item['train_pool_short']},{item['test_short']}"
            )
        (root / "粤语数量表.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for path, old in before.items():
            if sha256_file(Path(path)) != old:
                raise SystemExit(f"原中文jsonl被改动: {path}")
        for name, expect in CSV_HASH.items():
            if sha256_file(raw / name) != expect:
                raise SystemExit(f"{name} 写出后哈希变化")
    return {"summary": summary, "decisions": decisions, "kept_web": len(kept_web)}


def demo(repo: Path) -> None:
    rewriter = Rewriter()
    src = Path(r"D:\粤语AI语音导诊\scripts\yue_seed_annotations\triage_yue_seed_draft.jsonl")
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = []
    for row in rows[:8]:
        got = rewriter.to_yue(row["text_zh"])
        lines.append("ZH\n" + row["text_zh"])
        lines.append("RULE\n" + got)
        lines.append("SEED\n" + row["text_yue"])
        lines.append("---")
    Path(r"D:\粤语AI语音导诊\_tmp_yue_demo.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.demo:
        demo(repo)
    report = build(repo, args.raw, args.write)
    out = Path(r"D:\粤语AI语音导诊\_tmp_pc1_report.json")
    slim = {
        "summary": report["summary"],
        "kept_web": report["kept_web"],
        "decision_counts": dict(Counter(item["决定"] for item in report["decisions"])),
        "by_split": {
            "|".join("" if part is None else str(part) for part in key): value
            for key, value in Counter(
                (item["决定"], item["并入划分"], item["candidate_role"]) for item in report["decisions"]
            ).items()
        },
    }
    out.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
