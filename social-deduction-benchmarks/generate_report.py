#!/usr/bin/env python3
"""Generate a markdown research report from per-item JSON results.

Reads fields.yaml for the field/category structure and every *.json file in
the results directory, then writes report.md with a TOC (item name + year)
and detailed per-item sections grouped by field category.
"""

import json
import re
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
FIELDS_PATH = BASE / "fields.yaml"
OUTLINE_PATH = BASE / "outline.yaml"
RESULTS_DIR = BASE / "results"
REPORT_PATH = BASE / "report.md"

# JSON keys that are not data fields
INTERNAL_KEYS = {"uncertain", "_source_file"}

# Category multi-language / structural mapping (kept generic for compatibility)
CATEGORY_MAPPING = {
    "basic_info": ["basic_info", "Basic Info"],
    "game_coverage": ["game_coverage", "Game Coverage"],
    "agent_architecture": ["agent_architecture", "Agent Architecture"],
    "game_loop_design": ["game_loop_design", "Game Loop Design"],
    "evaluation": ["evaluation", "Evaluation"],
    "training_methodology": ["training_methodology", "Training Methodology"],
    "optimization": ["optimization", "Optimization"],
    "engineering": ["engineering", "Engineering"],
    "findings": ["findings", "Findings"],
}


def humanize(text: str) -> str:
    """Turn a snake_case identifier into a Title Case label."""
    return text.replace("_", " ").strip().title()


def github_anchor(heading: str) -> str:
    """Replicate GitHub's markdown heading -> anchor slug algorithm."""
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = slug.replace(" ", "-")
    return slug


def load_fields():
    """Return an ordered list of (category_name, description, [field_names])."""
    data = yaml.safe_load(FIELDS_PATH.read_text(encoding="utf-8"))
    cats = data.get("field_categories", [])
    ordered = []
    if isinstance(cats, list):
        for entry in cats:
            name = entry.get("category", "")
            desc = entry.get("description", "")
            fields = [f.get("name") for f in entry.get("fields", []) if f.get("name")]
            ordered.append((name, desc, fields))
    elif isinstance(cats, dict):  # dict-of-categories fallback
        for name, entry in cats.items():
            desc = entry.get("description", "")
            fields = [f.get("name") for f in entry.get("fields", []) if f.get("name")]
            ordered.append((name, desc, fields))
    return ordered


def get_value(item: dict, field: str):
    """Look up a field: top level, then any nested category dict."""
    if field in item:
        return item[field]
    for key in CATEGORY_MAPPING:
        sub = item.get(key)
        if isinstance(sub, dict) and field in sub:
            return sub[field]
    for value in item.values():
        if isinstance(value, dict) and field in value:
            return value[field]
    return None


def is_skippable(value, field: str, uncertain: set) -> bool:
    """Skip uncertain, empty, or flagged values."""
    if field in uncertain:
        return True
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        if not s or "[uncertain]" in s:
            return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False


def format_value(value) -> str:
    """Render a field value as readable markdown."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        lines = []
        for elem in value:
            if isinstance(elem, dict):
                lines.append(" | ".join(f"{k}: {v}" for k, v in elem.items()))
            else:
                lines.append(str(elem))
        if len(lines) <= 3 and all(len(line) < 60 for line in lines):
            return ", ".join(lines)
        return "<br>".join(f"- {line}" for line in lines)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def main():
    categories = load_fields()
    defined_fields = {f for _, _, fields in categories for f in fields}

    items = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"WARN: skipping unparseable {path.name}: {exc}")
            continue
        data["_source_file"] = path.name
        items.append(data)

    # Sort by year (unknown years last), then by name
    def sort_key(it):
        year = it.get("year")
        try:
            year_num = int(year)
        except (TypeError, ValueError):
            year_num = 9999
        return (year_num, str(it.get("name", "")))

    items.sort(key=sort_key)

    topic = ""
    if OUTLINE_PATH.exists():
        outline = yaml.safe_load(OUTLINE_PATH.read_text(encoding="utf-8"))
        topic = (outline.get("topic") or "").strip()

    out = []
    out.append("# Research Report")
    if topic:
        out.append("")
        out.append(f"> {topic}")
    out.append("")
    out.append(f"**Items:** {len(items)}  |  **Field categories:** {len(categories)}")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## Table of Contents")
    out.append("")

    used_anchors = {}
    headings = []
    for idx, item in enumerate(items, 1):
        name = str(item.get("name", item["_source_file"]))
        heading = f"{idx}. {name}"
        anchor = github_anchor(heading)
        # de-dup anchors
        if anchor in used_anchors:
            used_anchors[anchor] += 1
            anchor = f"{anchor}-{used_anchors[anchor]}"
        else:
            used_anchors[anchor] = 0
        headings.append((heading, anchor))
        year = item.get("year")
        year_str = str(year) if year not in (None, "") else "n/a"
        out.append(f"{idx}. [{name}](#{anchor}) — Year: {year_str}")

    out.append("")
    out.append("---")
    out.append("")
    out.append("## Detailed Findings")
    out.append("")

    for item, (heading, _anchor) in zip(items, headings):
        uncertain = set(item.get("uncertain") or [])
        out.append(f"## {heading}")
        out.append("")

        for cat_name, _desc, fields in categories:
            rows = []
            for field in fields:
                value = get_value(item, field)
                if is_skippable(value, field, uncertain):
                    continue
                rows.append((field, format_value(value)))
            if not rows:
                continue
            out.append(f"### {humanize(cat_name)}")
            out.append("")
            for field, rendered in rows:
                label = f"**{humanize(field)}:**"
                if "<br>" in rendered or len(rendered) > 100:
                    if out and out[-1] != "":
                        out.append("")
                    out.append(f"{label}")
                    out.append("")
                    out.append(f"> {rendered}")
                    out.append("")
                else:
                    out.append(f"- {label} {rendered}")
            out.append("")

        # Extra fields not defined in fields.yaml
        extras = []
        for key, value in item.items():
            if key in INTERNAL_KEYS or key in defined_fields:
                continue
            if key in CATEGORY_MAPPING:  # nested category top-level key
                continue
            if is_skippable(value, key, uncertain):
                continue
            extras.append((key, format_value(value)))
        if extras:
            out.append("### Other Info")
            out.append("")
            for key, rendered in extras:
                out.append(f"- **{humanize(key)}:** {rendered}")
            out.append("")

        # Uncertain fields listing
        if uncertain:
            out.append("### Uncertain Fields")
            out.append("")
            for field in sorted(uncertain):
                out.append(f"- {humanize(field)}")
            out.append("")

        out.append("---")
        out.append("")

    REPORT_PATH.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    print(f"Report written: {REPORT_PATH}")
    print(f"Items: {len(items)}  Categories: {len(categories)}")


if __name__ == "__main__":
    main()
