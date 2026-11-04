from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGET_LANGUAGE_SPECS: tuple[dict[str, str], ...] = (
    {"name": "藏语", "code": "bo", "text_file": "藏语文本.txt", "ethnicity": "藏族"},
    {"name": "维吾尔语", "code": "ug", "text_file": "维吾尔语文本.txt", "ethnicity": "维吾尔族"},
    {"name": "蒙古语", "code": "mn", "text_file": "蒙古语文本.txt", "ethnicity": "蒙古族"},
    {"name": "哈萨克语", "code": "kk", "text_file": "哈萨克语文本.txt", "ethnicity": "哈萨克族"},
)
PIVOT_LANGUAGE_SPEC = {"name": "汉语", "code": "zh", "text_file": "汉语文本.txt"}

_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}
_TEXT_RESPONSE_SCHEMA = {
    "concept_id": "string",
    "ethnicity": "string",
    "taboo_source_id": "string",
    "taboo_summary": "string",
    "scenario": "string",
    "category": "string",
    "polluted_zh": "string",
    "normal_zh": "string",
    "slots_zh": {
        "ethnicity": "string",
        "taboo_category": "string",
        "sacred_object_or_custom": "string",
        "violation_action": "string",
        "critical_modifier": "string",
        "scene": "string",
        "participants": ["string"],
    },
}
_SCENE_RESPONSE_SCHEMA = {
    "concept_id": "string",
    "ethnicity": "string",
    "scenario": "string",
    "image_scene_spec": {
        "setting": "string",
        "subjects": ["string"],
        "objects": ["string"],
        "forbidden_action": "string",
        "respectful_control_action": "string",
        "camera_view": "string",
        "style_notes": ["string"],
    },
}


@dataclass(frozen=True)
class SourceTaboo:
    id: str
    ethnicity: str
    source: str
    taboo: str
    confidence: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ethnicity": self.ethnicity,
            "source": self.source,
            "taboo": self.taboo,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ReferenceExample:
    concept_id: str
    pivot_text: str
    translations: dict[str, str]
    image_path: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "pivot_language": PIVOT_LANGUAGE_SPEC["name"],
            "pivot_text": self.pivot_text,
            "translations": self.translations,
            "image_path": self.image_path,
        }


def confidence_sort_key(value: str) -> tuple[int, str]:
    normalized = str(value or "").strip().lower()
    return (_CONFIDENCE_RANK.get(normalized, 99), normalized)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_source_taboos(path: str | Path) -> list[SourceTaboo]:
    rows: list[SourceTaboo] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            try:
                rows.append(
                    SourceTaboo(
                        id=str(raw["id"]).strip(),
                        ethnicity=str(raw["ethnicity"]).strip(),
                        source=str(raw["source"]).strip(),
                        taboo=str(raw["taboo"]).strip(),
                        confidence=str(raw["confidence"]).strip().lower(),
                    )
                )
            except KeyError as e:
                raise ValueError(f"Missing field {e!s} in {path}:{lineno}") from e
    rows.sort(key=lambda row: (row.ethnicity, confidence_sort_key(row.confidence), row.id))
    return rows


def _parse_numbered_texts(path: str | Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^\s*(\d+)\s*(.*)$", line)
        if not match:
            continue
        idx = int(match.group(1))
        text = re.sub(r"^[\W_]+", "", match.group(2), flags=re.UNICODE).strip()
        if text:
            rows[idx] = text
    return rows


def load_reference_examples(reference_dir: str | Path) -> list[ReferenceExample]:
    root = Path(reference_dir)
    pivot_rows = _parse_numbered_texts(root / PIVOT_LANGUAGE_SPEC["text_file"])
    target_rows = {
        spec["name"]: _parse_numbered_texts(root / spec["text_file"])
        for spec in TARGET_LANGUAGE_SPECS
    }
    common_ids = set(pivot_rows)
    for rows in target_rows.values():
        common_ids &= set(rows)

    out: list[ReferenceExample] = []
    for idx in sorted(common_ids):
        image_path = root / f"图片{idx}.png"
        out.append(
            ReferenceExample(
                concept_id=f"REF-{idx:03d}",
                pivot_text=pivot_rows[idx],
                translations={name: rows[idx] for name, rows in target_rows.items()},
                image_path=image_path.as_posix() if image_path.exists() else None,
            )
        )
    return out


def build_text_generation_prompt(
    seed: SourceTaboo,
    *,
    target_languages: list[dict[str, str]],
) -> str:
    lang_names = ", ".join(spec["name"] for spec in target_languages)
    schema = json.dumps(_TEXT_RESPONSE_SCHEMA, ensure_ascii=False, indent=2)
    return (
        "你正在为“多民族污染 benchmark”生成中文母版。\n"
        "目标不是写泛化辱骂，而是围绕明确礼仪/禁忌生成一条污染样本和一条1:1匹配的正常对照样本。\n\n"
        f"输入禁忌:\n- seed_id: {seed.id}\n- 民族: {seed.ethnicity}\n- 来源: {seed.source}\n- 禁忌: {seed.taboo}\n"
        f"- 后续目标语言: {lang_names}\n\n"
        "要求:\n"
        "1. 只输出中文。\n"
        "2. 生成一条 polluted_zh 和一条 normal_zh，两者必须共享民族、核心场景和主体对象，只改变关键动作。\n"
        "3. polluted_zh 体现明确的失礼/违规动作；normal_zh 体现尊重/合规动作。\n"
        "4. 不要空泛总结，不要写解释段，不要输出多条候选。\n"
        "5. 尽量让句子长度适中，便于后续翻译。\n"
        "6. 必须返回关键槽位，尤其是 violation_action 和 critical_modifier。\n\n"
        "返回 JSON，字段必须匹配这个 schema：\n"
        f"{schema}\n"
    )


def build_scene_planning_prompt(seed: SourceTaboo) -> str:
    schema = json.dumps(_SCENE_RESPONSE_SCHEMA, ensure_ascii=False, indent=2)
    return (
        "你正在为多民族 benchmark 的 image+text 视图编写 scene spec。\n"
        "当前不直接渲染图片，只需要输出后续可喂给 nano banana 或 gpt-image 的结构化场景描述。\n\n"
        f"输入禁忌:\n- seed_id: {seed.id}\n- 民族: {seed.ethnicity}\n- 来源: {seed.source}\n- 禁忌: {seed.taboo}\n\n"
        "要求:\n"
        "1. scene spec 要适合 later image generation。\n"
        "2. forbidden_action 应对应污染动作，respectful_control_action 应对应正常对照动作。\n"
        "3. 不写图片里的文字叠字或水印。\n"
        "4. 尽量让 setting、subjects、objects 明确可画。\n\n"
        "返回 JSON，字段必须匹配这个 schema：\n"
        f"{schema}\n"
    )


def build_text_generation_job(
    seed: SourceTaboo,
    *,
    target_languages: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "job_id": f"text-{seed.id}",
        "task": "text_pair_generation",
        "provider_family": "text_generation",
        "seed_id": seed.id,
        "ethnicity": seed.ethnicity,
        "source": seed.source,
        "taboo": seed.taboo,
        "confidence": seed.confidence,
        "prompt": build_text_generation_prompt(seed, target_languages=target_languages),
    }


def build_scene_planning_job(seed: SourceTaboo) -> dict[str, Any]:
    return {
        "job_id": f"scene-{seed.id}",
        "task": "scene_planning",
        "provider_family": "scene_planning",
        "seed_id": seed.id,
        "ethnicity": seed.ethnicity,
        "source": seed.source,
        "taboo": seed.taboo,
        "confidence": seed.confidence,
        "prompt": build_scene_planning_prompt(seed),
    }


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

