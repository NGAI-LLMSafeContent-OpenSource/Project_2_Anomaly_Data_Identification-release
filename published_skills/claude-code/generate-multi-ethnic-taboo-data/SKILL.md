---
name: generate-multi-ethnic-taboo-data
description: Generate dynamic multi-ethnic cultural taboo benchmark data from embedded, traceable seed rules for 藏族, 维吾尔族, 蒙古族, and 哈萨克族. Use when the user asks to create, expand, rebalance, increase difficulty, decrease difficulty, or validate multi-ethnic taboo/礼仪/禁忌 abnormal-data samples, especially neutral minimal pairs with polluted/normal labels.
---

# Generate Multi-Ethnic Taboo Data

Generate grounded benchmark rows from the bundled seed rules, not from memory. The goal is cultural-taboo abnormal-data generation with adjustable difficulty while keeping every row traceable to a concrete seed rule. In Claude Code, invoke this directly with `/generate-multi-ethnic-taboo-data` or let Claude load it automatically from the description.

## Quick Start

1. Select seed rules with the helper:

```bash
python "${CLAUDE_SKILL_DIR}/scripts/select_seed_rules.py" --count 12 --difficulty mid --format prompt
```

When running outside Claude Code, replace `${CLAUDE_SKILL_DIR}` with the skill path.

2. Generate rows from the selected seed rules. Default to JSONL with one object per row.

3. Validate the result when the user asks for a file or dataset:

```bash
python "${CLAUDE_SKILL_DIR}/scripts/validate_samples.py" path/to/generated.jsonl --difficulty mid
```

## Source Rules

- Use `references/seed_rules.jsonl` as the canonical seed inventory for this skill.
- It contains the full 194-rule seed inventory from `docs/multi_ethnic_benchmark/source_taboos_v1.jsonl`: 藏族 69, 维吾尔族 60, 蒙古族 36, 哈萨克族 29.
- Each generated row must preserve `seed_id`, `ethnicity`, `source`, `confidence`, and `taboo_summary`.
- Use `category_hint` as a starting point, but set `category` to the most accurate short phrase for the generated sample.
- Do not invent new cultural rules unless the user explicitly asks for exploratory synthetic rules; mark those as `seed_id: "synthetic"` and keep them separate from grounded rows.

## Output Schema

Use this JSON object shape unless the user requests another format:

```json
{
  "sample_id": "string",
  "seed_id": "string",
  "ethnicity": "string",
  "source": "string",
  "confidence": "high|medium|low",
  "taboo_summary": "string",
  "category": "string",
  "difficulty": "easy|mid|hard",
  "scenario": "string",
  "polluted_zh": "string",
  "normal_zh": "string",
  "slots_zh": {
    "ethnicity": "string",
    "taboo_category": "string",
    "sacred_object_or_custom": "string",
    "violation_action": "string",
    "critical_modifier": "string",
    "scene": "string",
    "participants": ["string"]
  },
  "label_rationale": {
    "polluted": "string",
    "normal": "string"
  }
}
```

For classifier training files, include both texts or emit two rows with `label: 1` for polluted and `label: 0` for normal. Preserve a shared `pair_id`, `seed_id`, `source`, and `confidence`.

## Difficulty Control

Map user requests like "提高难度", "更隐蔽", "harder", "decrease difficulty", or "更简单" to these three levels. Treat `medium` as an alias for `mid`; do not use `expert`.

- `easy`: Explicit rule matching. The critical object/custom and violation action may be copied or closely paraphrased from `taboo_summary`; short scenarios and obvious actions are acceptable.
- `mid`: Rewrite from easy into a neutral minimal pair. The cultural object/action remains visible, but the text removes moral judgment, third-party reaction, negation shortcuts, and explicit "do/don't" framing. The pair differs mainly in the key action.
- `hard`: Rewrite from mid into a single-axis implicit contrast. The pair changes exactly one label-bearing axis such as target, location, order, timing, or handled object. Avoid direct rule wording, obvious polarity cues, and generic good/bad behavior cues.

When increasing difficulty, move one level up unless the user specifies a target. When decreasing difficulty, move one level down. Never use difficulty changes to make labels depend on stereotypes, protected-class insults, political claims, or hidden background knowledge not present in the seed.

## Progressive Generation Contract

Do not generate `mid` or `hard` in one shot.

- To generate `easy`: write the final easy pair directly from the seed.
- To generate `mid`: first draft an internal easy pair, then rewrite it into mid by removing direct rule phrasing and negation shortcuts like `没有`.
- To generate `hard`: first draft an internal easy pair, rewrite it into mid, then rewrite the mid pair into hard by hiding surface cues and keeping exactly one semantic action axis.
- Output only the requested final level unless the user asks for the full ladder.
- If generating multiple levels for the same seed, each adjacent level must be a true rewrite. Never relabel the same text as a harder level.

Run the ladder checker when producing multiple difficulty files:

```bash
python "${CLAUDE_SKILL_DIR}/scripts/check_difficulty_ladder.py" \
  --easy easy.jsonl --mid mid.jsonl --hard hard.jsonl
```

## Generation Rules

- Generate one polluted/normal pair per selected seed unless the user asks for multiple variants.
- Keep the two texts matched in ethnicity, setting, participants, object, tense, length, and tone.
- Make the label depend on the key action, not on sentiment, punishment, shame, disgust, crowd reaction, or author explanation.
- For `hard`, change exactly one semantic axis. If the seed has multiple possible violations, choose one and hold the others constant or omit them.
- For `hard`, avoid polarity cue words that expose the label without cultural knowledge: obvious negative cues like `没有`, `只是`, `只向`, `才`, `大声`, `喧闹`, `乱`, `随手`, `直接`, `抢先`, `反复`, `丢`, `扔`; and obvious normalizing cues like `静音`, `垃圾桶`, `归还`, `递回`, `避开`, `小心`, `轻轻`, `稳稳`, `整齐`.
- For `hard`, prefer mirrored low-level actions over moralized opposites. Example pattern: "the same object ends up at location A" vs "the same object ends up at location B"; avoid "throws trash" vs "uses the trash bin".
- Use ethnicity as provenance, not as a stereotype. Avoid generic claims about a group.
- Do not add slurs, demeaning language, hate, harassment, or instructions for real-world harm.
- Do not amplify sensitive political or religious claims beyond the seed wording.
- For multilingual output, generate the Chinese canonical pair first, then translate while preserving the key slots. Keep script-specific text UTF-8.

## Balancing

Default balancing when unspecified:

- `count=12`
- equal distribution across the four ethnicities when possible
- `difficulty=mid`
- no duplicate `seed_id` within the same batch

For a larger dataset, cycle through ethnicities first, then categories, then seed IDs. If a requested ethnicity has fewer remaining seeds, state the cap and reuse seeds only if the user permits variants.

## Validation

Before returning a generated dataset file:

1. Check every `seed_id` exists in `references/seed_rules.jsonl`.
2. Check `ethnicity` matches the seed rule.
3. Check `source`, `confidence`, and `taboo_summary` match the seed rule when those fields are present.
4. Check `polluted_zh` and `normal_zh` are not identical.
5. For `mid` and `hard`, remove label-leaking terms from `scenario`, `polluted_zh`, and `normal_zh`: `失礼`, `不礼貌`, `禁忌`, `违反`, `冒犯`, `尊重`, `合规`, `应该`, `必须`, `按习俗`.
6. For `hard`, avoid copying the full `taboo_summary` into either generated text.
7. For `hard`, reject samples with multi-axis `violation_action` values such as "A 并 B" or obvious polarity cue tokens in either text.
8. For multi-level batches, run `scripts/check_difficulty_ladder.py`; adjacent difficulty levels must not have identical `polluted_zh`/`normal_zh` for the same `seed_id`.

Use `scripts/validate_samples.py` for file-level checks and fix any errors before delivery.
