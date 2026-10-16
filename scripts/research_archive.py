#!/usr/bin/env python3
"""Build/search a local research asset index without copying or loading assets.

Curated scientific judgments live in reports/research_archive/*.json. Inventory
status fields are execution metadata, never evidence of a valid scientific claim.
No checkpoints are deserialized, no large files are hashed, and no APIs are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reports/research_archive"
AREAS = ["configs", "reports", "scripts", "src", "tests", "models", "checkpoints", "data", "runs", "vendor"]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules"}
CURATED = ["mechanisms.json", "models.json", "evidence.json", "papers.json", "lessons.json", "next_experiment_gate.json"]
SECRET = re.compile(r"sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,}|AQ\.[A-Za-z0-9._-]{20,}")
KEY_FILES = {"summary.json", "status.json", "results.json", "analysis.json", "plan.json", "protocol.json", "provenance.json", "statistics.json"}


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def execution_status(path, size):
    if path.name not in KEY_FILES or size > 512_000:
        return None
    try:
        value = json.loads(path.read_text()).get("status")
        # Only a small machine status, never free-form objectives/credentials.
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_. /-]{1,100}", value):
            return value
    except (ValueError, OSError, AttributeError, UnicodeError):
        pass
    return None


def iter_assets():
    for area in AREAS:
        base = ROOT / area
        if not base.exists():
            continue
        for directory, children, files in os.walk(base, followlinks=False):
            parent = Path(directory)
            children[:] = sorted(c for c in children if c not in SKIP_DIRS
                                 and parent / c != ARCHIVE)
            symlink_dirs = [c for c in children if (parent / c).is_symlink()]
            for name in sorted(files + symlink_dirs):
                path = parent / name
                if path.suffix == ".pyc":
                    continue
                try:
                    stat = path.lstat()
                    record = {"path": path.relative_to(ROOT).as_posix(), "area": area,
                              "bytes": stat.st_size, "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
                    if path.is_symlink():
                        record.update(kind="symlink", target=os.readlink(path), target_exists=path.exists())
                    else:
                        record["kind"] = "file"
                        status = execution_status(path, stat.st_size)
                        if status is not None:
                            record["execution_status_only"] = status
                    yield record
                except OSError as error:
                    yield {"path": path.relative_to(ROOT).as_posix(), "area": area,
                           "kind": "unreadable_metadata", "error_type": type(error).__name__}
    for path in sorted(ROOT.iterdir()):
        if path.is_file() and path.suffix.lower() in {".pdf", ".tex", ".md", ".bib", ".html"}:
            yield {"path": path.name, "area": "root_documents", "kind": "file", "bytes": path.stat().st_size}


def curated_data():
    data = {}
    for name in CURATED:
        path = ARCHIVE / name
        if not path.exists():
            raise FileNotFoundError(f"Required curated registry is not ready: {name}")
        raw = path.read_text()
        if SECRET.search(raw):
            raise ValueError(f"Credential-like text in curated registry: {name}; do not publish")
        data[name.removesuffix(".json")] = json.loads(raw)
    return data


def write_index(data, inventory):
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    # Link only exact, present project paths. Descriptions of unavailable files
    # and glob patterns remain plain text; external caches stay explicit.
    links = {}
    def collect_links(value):
        if isinstance(value, dict):
            for child in value.values():
                collect_links(child)
        elif isinstance(value, list):
            for child in value:
                collect_links(child)
        elif isinstance(value, str) and len(value) < 500 and "\n" not in value:
            path = Path(value)
            if not path.is_absolute() and (path.parts[0:1] and
                    (path.parts[0] in AREAS or value.endswith(".pdf"))):
                candidate = ROOT / path
                if candidate.exists() and candidate.resolve().is_relative_to(ROOT):
                    links[value] = "../../" + value
    collect_links(data)
    link_payload = json.dumps(links, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    counts = " · ".join(f"{html.escape(k)} {v}" for k, v in inventory["file_count_by_area"].items())
    page = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>World Model Security 研究资产与经验归档</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1180px;margin:36px auto;padding:0 22px;color:#182333}h1{font-size:28px}h2{margin-top:32px}a{color:#17629c}input{width:100%;box-sizing:border-box;padding:12px;font:inherit;border:1px solid #9caec0;border-radius:6px}.notice{background:#fff4d7;border-left:4px solid #c28d26;padding:14px}.small{font-size:13px;color:#586477}details{border:1px solid #d8e0e7;border-radius:6px;margin:10px 0;padding:12px}summary{cursor:pointer;font-weight:600}dl{margin:8px 0}dt{font-weight:600;margin-top:8px}dd{margin-left:16px;overflow-wrap:anywhere}li{margin:5px 0}.entry{background:white}nav{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}code{background:#edf2f7;padding:2px 4px}pre{white-space:pre-wrap}.hidden{display:none}</style>
<body><h1>World Model Security + Agent：研究资产与失败经验</h1>
<p class="notice"><strong>当前尚无达到论文目标的完整安全证据链。</strong> 预测变化、运行完成、隔离后恒等及主观 novelty 评分都不能替代真实危害、有效防御或独立泛化。先查下方机制登记，再决定是否开展新实验。</p>
<p>原目录继续保存论文、代码、日志、权重与缓存；本归档提供统一索引和有效性说明，不复制大模型、不修改旧结果。历史结论的适用范围以当前证据登记为准。原始 <code>status=completed</code> 只表示执行结束，不表示科学主张成立。</p>
<nav><a href="#mechanisms">机制与止损记录</a><a href="#models">模型与决策路径</a><a href="#evidence">证据链</a><a href="#papers">论文与文献</a><a href="#lessons">开发经验</a><a href="#next_experiment_gate">下一实验门</a><a href="../research_progress.html">完整研究报告</a></nav>
<label for="search">搜索机制、模型、失败原因或文件路径</label><input id="search" type="search" placeholder="例如 Goal-Swap、TD-MPC2、codec、checkpoint">
<p class="small">__COUNTS__</p><p>完整文件位置索引：<a href="assets.jsonl">assets.jsonl</a>；运行目录索引：<a href="runs.json">runs.json</a>；<a href="inventory.json">覆盖范围与数量</a>。离线查询：<code>python scripts/research_archive.py find WorldAgen</code>。</p>
<main id="registries"></main><p class="small">生成时间：__TIME__。目录扫描不读取凭据内容、不反序列化 checkpoint、不进行大文件散列审计。</p>
<script id="archive-data" type="application/json">__DATA__</script>
<script id="archive-links" type="application/json">__LINKS__</script>
<script>
const data=JSON.parse(document.getElementById('archive-data').textContent);
const localLinks=JSON.parse(document.getElementById('archive-links').textContent);
const labels={mechanisms:'机制、威胁模型与失败记录',models:'模型、Checkpoint、Agent 接口与缓存',evidence:'当前证据链与引用范围',papers:'论文、报告与文献来源',lessons:'可复用开发经验',next_experiment_gate:'下一项决定性实验的入口条件'};
function render(v){if(v===null)return document.createTextNode('未提供');if(Array.isArray(v)){const u=document.createElement('ul');for(const x of v){const l=document.createElement('li');l.append(render(x));u.append(l)}return u}if(typeof v==='object'){const d=document.createElement('dl');for(const[k,x]of Object.entries(v)){const t=document.createElement('dt');t.textContent=k;const e=document.createElement('dd');e.append(render(x));d.append(t,e)}return d}const s=String(v);if(Object.hasOwn(localLinks,s)){const a=document.createElement('a');a.textContent=s;a.href=localLinks[s];return a}if(/^https?:\/\/[^\s]+$/.test(s)){const a=document.createElement('a');a.textContent=s;a.href=s;return a}return document.createTextNode(s)}
for(const[k,v]of Object.entries(data)){const sec=document.createElement('section');sec.id=k;const h=document.createElement('h2');h.textContent=labels[k]||k;sec.append(h);const link=document.createElement('a');link.href=k+'.json';link.textContent='机器可读登记';sec.append(link);let entries=[];let metadata={};if(Array.isArray(v))entries=v;else{for(const[n,x]of Object.entries(v)){if(Array.isArray(x)&&x.some(y=>y&&typeof y==='object'))entries.push(...x.map(y=>({registry_group:n,...y})));else metadata[n]=x}}if(Object.keys(metadata).length){const m=document.createElement('details');const q=document.createElement('summary');q.textContent='范围与解释';m.append(q,render(metadata));sec.append(m)}for(const e of entries){const d=document.createElement('details');d.className='entry';d.dataset.search=JSON.stringify(e).toLowerCase();const q=document.createElement('summary');q.textContent=e.name||e.title||e.id||e.path||e.model||e.registry_group||'记录';d.append(q,render(e));sec.append(d)}document.getElementById('registries').append(sec)}
document.getElementById('search').addEventListener('input',ev=>{const q=ev.target.value.toLowerCase().trim();for(const e of document.querySelectorAll('.entry')){const yes=!q||e.dataset.search.includes(q);e.classList.toggle('hidden',!yes);if(q&&yes)e.open=true}});
</script></body></html>'''
    page = page.replace("__COUNTS__", counts).replace("__TIME__", inventory["built_at_utc"]).replace("__DATA__", payload).replace("__LINKS__", link_payload)
    (ARCHIVE / "index.html").write_text(page)


def build():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    data = curated_data()
    counts, sizes, runs = Counter(), Counter(), {}
    destination = ARCHIVE / "assets.jsonl"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for record in iter_assets():
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            area = record["area"]; counts[area] += 1; sizes[area] += record.get("bytes", 0)
            parts = Path(record["path"]).parts
            if area == "runs" and len(parts) >= 3:
                run = runs.setdefault(parts[1], {"path": "runs/" + parts[1], "file_count": 0, "bytes": 0, "key_artifacts": []})
                run["file_count"] += 1; run["bytes"] += record.get("bytes", 0)
                if parts[-1] in KEY_FILES:
                    run["key_artifacts"].append({k: v for k, v in record.items() if k in {"path", "bytes", "execution_status_only"}})
    temporary.replace(destination)
    environments = [{"path": p.name, "kind": "environment_or_dependency_directory", "contents_not_rescanned": True}
                    for p in sorted(ROOT.iterdir()) if p.is_dir() and (p.name.startswith(".venv-") or p.name in {".deps", ".libero"})]
    inventory = {"schema_version": 1, "built_at_utc": datetime.now(timezone.utc).isoformat(),
                 "root": str(ROOT), "archive_mode": "in_place_index_original_assets_preserved",
                 "file_count_by_area": dict(counts), "bytes_by_area_no_symlink_target_sizes": dict(sizes),
                 "total_indexed_entries": sum(counts.values()), "run_directories": len(runs),
                 "areas": AREAS, "excluded_directory_names": sorted(SKIP_DIRS),
                 "self_archive_excluded_from_scan": True, "symlink_targets_not_traversed": True,
                 "environment_directories": environments,
                 "scope": "Metadata of local project assets, including vendor source. Files stay in place; external symlink targets are referenced. No claim that every old experiment is scientifically valid.",
                 "curated_registries": CURATED, "model_loads": 0, "environment_steps": 0, "api_requests": 0,
                 "checkpoint_hash_passes": 0}
    dump(ARCHIVE / "runs.json", {"execution_metadata_only": True, "runs": list(runs.values())})
    dump(ARCHIVE / "inventory.json", inventory)
    write_index(data, inventory)
    print(json.dumps({"index": str(ARCHIVE / "index.html"), "entries": sum(counts.values()), "runs": len(runs), "areas": dict(counts)}, ensure_ascii=False))


def find(query, area, limit):
    words = query.casefold().split()
    found = 0
    with (ARCHIVE / "assets.jsonl").open() as handle:
        for line in handle:
            item = json.loads(line)
            if area and item["area"] != area:
                continue
            if all(word in item["path"].casefold() for word in words):
                print(json.dumps(item, ensure_ascii=False)); found += 1
                if found >= limit:
                    break
    print(json.dumps({"shown": found, "limit": limit, "scientific_verdict": "Consult curated mechanisms/evidence registries"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    search = sub.add_parser("find"); search.add_argument("query"); search.add_argument("--area", choices=AREAS)
    search.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    if args.command == "build":
        build()
    else:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        find(args.query, args.area, args.limit)


if __name__ == "__main__":
    main()
