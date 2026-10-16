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
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reports/research_archive"
AREAS = ["configs", "reports", "scripts", "src", "tests", "models", "checkpoints", "data", "runs", "vendor", "reproducibility"]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".secrets", "cpu_deps"}
CURATED = ["mechanisms.json", "models.json", "evidence.json", "papers.json", "lessons.json", "next_experiment_gate.json", "resource_recipes.json", "reuse_guide.json"]
SECRET = re.compile(r"sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,}|AQ\.[A-Za-z0-9._-]{20,}|github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{24,}")
PRIVATE_NAMES = {"api_endpoints_status.json", "api.json", "api.local.json", "gemini.json", "xinlicloud.json", "credentials.json", "credentials.local.json", "secrets.json", "api_keys.json"}
KEY_FILES = {"summary.json", "status.json", "results.json", "analysis.json", "plan.json", "protocol.json", "provenance.json", "statistics.json"}
GENERATED_PATHS = {"reports/research_archive/" + name for name in ("assets.jsonl", "runs.json", "catalog.json", "coverage.json", "inventory.json", "index.html")}


def private_path(path):
    return path.name in PRIVATE_NAMES or path.suffix in {".pem", ".key", ".p12", ".pfx"} or any(p.startswith(".env") or p == ".secrets" for p in path.parts)


def git_inventory():
    """Local index membership, not a claim that current bytes were published."""
    tracked, submodules = set(), {}
    try:
        raw = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--stage", "-z"], stderr=subprocess.DEVNULL)
        for row in raw.split(b"\0"):
            if not row:
                continue
            metadata, path = row.split(b"\t", 1)
            mode, oid, _ = metadata.decode().split()
            path = path.decode()
            if mode == "160000":
                submodules[path] = oid
            else:
                tracked.add(path)
        return tracked, submodules, True
    except (OSError, subprocess.CalledProcessError):
        return tracked, submodules, False


def storage_class(path, git):
    tracked, submodules, available = git
    if not available:
        return "git_status_unavailable"
    if path in tracked:
        return "git_tracked_path"
    if any(path == parent or path.startswith(parent + "/") for parent in submodules):
        return "pinned_upstream_submodule"
    return "local_only_path"


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
            children[:] = sorted(c for c in children if c not in SKIP_DIRS and not c.startswith(".venv")
                                 and parent / c != ARCHIVE)
            symlink_dirs = [c for c in children if (parent / c).is_symlink()]
            for name in sorted(files + symlink_dirs):
                path = parent / name
                if path.suffix in {".pyc", ".nbc", ".nbi"} or private_path(path):
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


PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9_/:.-])(?:" + "|".join(AREAS) + r")/[A-Za-z0-9_./+@%=-]+")


def local_references(value):
    """Explicit local path mentions only; never infer a scientific relationship."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.update(local_references(key))
            found.update(local_references(item))
    elif isinstance(value, list):
        for item in value:
            found.update(local_references(item))
    elif isinstance(value, str):
        text = value.replace(str(ROOT) + "/", "")
        for match in PATH_PATTERN.finditer(text):
            path = match.group().rstrip(".,/")
            # Glob/templated mentions remain in the original registry; a
            # truncated literal prefix must not falsely cover every child run.
            following = text[match.end():match.end() + 1]
            if following and following in "*?{[":
                continue
            p = Path(path)
            if ".." not in p.parts and not private_path(p):
                found.add(path)
        if "\n" not in text and len(text) < 500:
            p = Path(text)
            if len(p.parts) == 1 and p.suffix.lower() in {".md", ".pdf", ".tex", ".html"} and (ROOT / p).is_file():
                found.add(text)
    return found


def registry_records(data):
    for registry, document in data.items():
        if registry == "next_experiment_gate":
            yield registry, "current", 0, {"id": "current_execution_gate", "name": "当前研究状态与下一步入口", **document}
            continue
        groups = document.items() if isinstance(document, dict) else [("entries", document)]
        for group, items in groups:
            if isinstance(items, list):
                for index, item in enumerate(items):
                    if isinstance(item, dict):
                        yield registry, group, index, item


def source_reference_graph(assets):
    """Follow explicit mentions in small structured provenance, not raw traces."""
    graph = {}
    for item in assets:
        path = Path(item["path"])
        is_source = item["area"] in {"reports", "configs"} or (item["area"] == "runs" and path.name in KEY_FILES)
        limit = 4_000_000 if item["area"] in {"reports", "configs"} else 512_000
        if not is_source or path.suffix != ".json" or item.get("bytes", 0) > limit or item["kind"] != "file":
            continue
        try:
            value = json.loads((ROOT / path).read_text())
            refs = {p for p in local_references(value) if (ROOT / p).exists() and p != path.as_posix()}
            if refs:
                graph[path.as_posix()] = sorted(refs)
        except (OSError, UnicodeError, ValueError):
            continue
    return graph


def make_catalog(data, assets, git):
    entries = []
    owners = {}
    asset_map = {item["path"]: item for item in assets}
    source_graph = source_reference_graph(assets)
    for registry, group, index, item in registry_records(data):
        uid = f"{registry}:{group}:{index}"
        references = local_references(item)
        if registry == "mechanisms" and group == "entries":
            references.update(path for path, owner in data["mechanisms"].get("config_to_entry", {}).items()
                              if owner == item.get("id") and not private_path(Path(path)))
        references = sorted(references)
        status = item.get("results_status", item.get("status", item.get("readiness", item.get("verification_status", "not_recorded"))))
        dependencies = {}
        frontier = {path: [path] for path in references}
        for _ in range(2):
            following = {}
            for parent, chain in frontier.items():
                for child in source_graph.get(parent, []):
                    if child in references or child in dependencies:
                        continue
                    dependencies[child] = chain
                    following[child] = chain + [child]
            frontier = following
        entry = {"uid": uid, "registry": registry, "group": group,
                 "id": item.get("id", item.get("model_id")), "title": item.get("name", item.get("title", item.get("id", item.get("path", group)))),
                 "status": status, "local_references": references,
                 "source_dependencies": [{"path": path, "via": via} for path, via in sorted(dependencies.items())],
                 "missing_local_references": [p for p in references if p not in GENERATED_PATHS and not (ROOT / p).exists()],
                 "search_text": json.dumps(item, ensure_ascii=False).casefold(), "related_records": []}
        entries.append(entry)
        for path in references + list(dependencies):
            owners.setdefault(path, []).append(uid)
    by_uid = {entry["uid"]: entry for entry in entries}
    for path, linked in owners.items():
        # Shared overview pages, registries, and broad directories are navigation,
        # not discriminative provenance. Do not create misleading all-to-all links.
        if path == "reports/research_progress.html" or path.startswith("reports/research_archive/") or path not in asset_map:
            continue
        if len(linked) > 12:
            continue
        for uid in linked:
            for other in linked:
                if other != uid:
                    by_uid[uid]["related_records"].append({"uid": other, "via": path, "relation": "shared_explicit_path_or_source_reference_chain_only"})
    for entry in entries:
        grouped = {}
        for link in entry["related_records"]:
            grouped.setdefault(link["uid"], set()).add(link["via"])
        entry["related_records"] = [{"uid": uid, "shared_path_count": len(paths), "via_examples": sorted(paths)[:3],
                                     "relation": "shared_explicit_path_or_source_reference_chain_only"}
                                    for uid, paths in sorted(grouped.items())]
    return {"schema_version": 1, "association_policy": "Explicit paths plus at most two hops through structured report/config/run-summary references; dependencies retain their provenance chain. Shared source does not establish causal support, reproduction, or scientific validity. Ordinal uid is a navigation key; use registry + group + id for durable named references.",
            "entry_count": len(entries), "entries": entries}, owners


def coverage_report(data, assets, runs, owners, git):
    def references_for(path):
        matches = set(owners.get(path, []))
        for parent in Path(path).parents:
            prefix = parent.as_posix()
            if len(parent.parts) >= 2:
                matches.update(owners.get(prefix, []))
        return sorted(matches)
    run_links = {}
    for path, linked in owners.items():
        parts = Path(path).parts
        if len(parts) >= 2 and parts[0] == "runs":
            run_links.setdefault("/".join(parts[:2]), set()).update(linked)
    unlinked_runs, linked_runs = [], 0
    for run in runs.values():
        run["curated_record_refs"] = sorted(run_links.get(run["path"], []))
        run["scientific_status"] = "consult_linked_curated_records" if run["curated_record_refs"] else "not_individually_curated"
        if run["curated_record_refs"]:
            linked_runs += 1
        else:
            unlinked_runs.append(run["path"])
    pending = []
    by_area = Counter()
    storage = Counter()
    for item in assets:
        storage[item["storage_class"]] += 1
        if item["area"] not in {"configs", "reports", "scripts", "src", "tests", "data", "reproducibility"}:
            continue
        if not references_for(item["path"]):
            by_area[item["area"]] += 1
            pending.append({"path": item["path"], "area": item["area"], "status": "indexed_without_explicit_curated_reference"})
    missing = [{"path": path, "referenced_by": linked, "classification": "unresolved_path_like_mention_not_confirmed_missing_asset"}
               for path, linked in sorted(owners.items()) if path not in GENERATED_PATHS and not (ROOT / path).exists()]
    return {"schema_version": 1, "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Filesystem and explicit-reference coverage, including up to two structured-source hops. No result is newly validated; unresolved path-like mentions can be prose, remote paths, templates, or historical assets and require manual classification. Unlinked assets remain searchable and are not automatically discarded or marked failed.",
            "total_indexed_assets": len(assets), "total_run_directories": len(runs),
            "runs_with_curated_references": linked_runs, "runs_without_individual_curated_references": unlinked_runs,
            "unlinked_assets_by_area": dict(by_area), "unlinked_assets": pending,
            "missing_literal_path_mentions": missing, "storage_counts": dict(storage),
            "git_membership_only_not_remote_content_verification": True,
            "private_api_configuration_excluded": True,
            "validation_activity": {"model_loads": 0, "environment_steps": 0, "api_requests": 0, "checkpoint_hash_passes": 0}}


def write_index(data, inventory, catalog, coverage):
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
            if private_path(path):
                return
            if not path.is_absolute() and (path.parts[0:1] and
                    (path.parts[0] in AREAS or value.endswith(".pdf"))):
                candidate = ROOT / path
                if candidate.exists() and candidate.resolve().is_relative_to(ROOT):
                    links[value] = "../../" + value
    compact_catalog = {e["uid"]: {"title": e["title"], "related_records": e["related_records"][:30],
                                  "related_record_count": len(e["related_records"]),
                                  "source_dependencies": e["source_dependencies"][:30],
                                  "source_dependency_count": len(e["source_dependencies"])} for e in catalog["entries"]}
    collect_links(data)
    collect_links(compact_catalog)
    link_payload = json.dumps(links, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    catalog_payload = json.dumps(compact_catalog, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    counts = " · ".join(f"{html.escape(k)} {v}" for k, v in inventory["file_count_by_area"].items())
    page = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>World Model Security 研究资产与经验归档</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1180px;margin:36px auto;padding:0 22px;color:#182333}h1{font-size:28px}h2{margin-top:32px}a{color:#17629c}input{width:100%;box-sizing:border-box;padding:12px;font:inherit;border:1px solid #9caec0;border-radius:6px}.notice{background:#fff4d7;border-left:4px solid #c28d26;padding:14px}.small{font-size:13px;color:#586477}details{border:1px solid #d8e0e7;border-radius:6px;margin:10px 0;padding:12px}summary{cursor:pointer;font-weight:600}dl{margin:8px 0}dt{font-weight:600;margin-top:8px}dd{margin-left:16px;overflow-wrap:anywhere}li{margin:5px 0}.entry{background:white}nav{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}code{background:#edf2f7;padding:2px 4px}pre{white-space:pre-wrap}.hidden{display:none}</style>
<body><h1>World Model Security + Agent：研究资产与失败经验</h1>
<p class="notice"><strong>当前尚无达到论文目标的完整安全证据链。</strong> 预测变化、运行完成、隔离后恒等及主观 novelty 评分都不能替代真实危害、有效防御或独立泛化。先查下方机制登记，再决定是否开展新实验。</p>
<p>原目录继续保存论文、代码、日志、权重与缓存；本归档提供统一索引和有效性说明，不复制大模型、不修改旧结果。历史结论的适用范围以当前证据登记为准。原始 <code>status=completed</code> 只表示执行结束，不表示科学主张成立。</p>
<nav><a href="#evidence">当前可信证据</a><a href="#resource_recipes">模型复用配方</a><a href="#reuse_guide">流程、指标与 Baseline</a><a href="#mechanisms">机制与止损记录</a><a href="#models">模型与决策路径</a><a href="#papers">论文与文献</a><a href="#lessons">开发经验</a><a href="#next_experiment_gate">下一实验门</a><a href="../research_progress.html">完整研究报告</a></nav>
<p class="notice">__COVERAGE__<br>文件索引覆盖不等于逐条科学复核；缺少结论关联的旧运行仍完整保留。先看具体修正报告及证据卡，再引用历史文稿。</p>
<label for="search">搜索机制、模型、失败原因或文件路径</label><input id="search" type="search" placeholder="例如 Goal-Swap、TD-MPC2、codec、checkpoint">
<p><label>分类 <select id="registry-filter"><option value="">全部分类</option></select></label> <label>机制结果 <select id="status-filter"><option value="">不限</option></select></label> <span id="search-count" class="small"></span></p>
<p class="small">__COUNTS__</p><p>完整文件位置索引：<a href="assets.jsonl">assets.jsonl</a>；运行目录索引：<a href="runs.json">runs.json</a>；<a href="inventory.json">覆盖范围与数量</a>。离线查询：<code>python scripts/research_archive.py find WorldAgen</code>。</p>
<p>知识卡检索：<code>python scripts/research_archive.py search "真实 效用"</code>；<a href="catalog.json">卡片关联目录</a>；<a href="coverage.json">待判定路径提及与待整理队列</a>。<strong>Git 跟踪路径不表示本次文件内容已推送；本机权重、数据与外部缓存见模型复用配方。</strong></p>
<main id="registries"></main><p class="small">生成时间：__TIME__。目录扫描不读取凭据内容、不反序列化 checkpoint、不进行大文件散列审计。</p>
<script id="archive-data" type="application/json">__DATA__</script>
<script id="archive-links" type="application/json">__LINKS__</script>
<script id="archive-catalog" type="application/json">__CATALOG__</script>
<script>
const data=JSON.parse(document.getElementById('archive-data').textContent);
const localLinks=JSON.parse(document.getElementById('archive-links').textContent);
const catalog=JSON.parse(document.getElementById('archive-catalog').textContent);
const labels={mechanisms:'机制、威胁模型与失败记录',models:'模型、Checkpoint、Agent 接口与缓存',evidence:'当前证据链与引用范围',papers:'论文、报告与文献来源',lessons:'可复用开发经验',next_experiment_gate:'下一项决定性实验的入口条件',resource_recipes:'模型、环境与缓存复用配方',reuse_guide:'实验流程、评价指标、Baseline 与维护方法'};
function render(v){if(v===null)return document.createTextNode('未提供');if(Array.isArray(v)){const u=document.createElement('ul');for(const x of v){const l=document.createElement('li');l.append(render(x));u.append(l)}return u}if(typeof v==='object'){const d=document.createElement('dl');for(const[k,x]of Object.entries(v)){const t=document.createElement('dt');t.textContent=k;const e=document.createElement('dd');e.append(render(x));d.append(t,e)}return d}const s=String(v);if(Object.hasOwn(localLinks,s)){const a=document.createElement('a');a.textContent=s;a.href=localLinks[s];return a}if(/^https?:\/\/[^\s]+$/.test(s)){const a=document.createElement('a');a.textContent=s;a.href=s;return a}return document.createTextNode(s)}
const order=['evidence','resource_recipes','reuse_guide','mechanisms','models','papers','lessons','next_experiment_gate'];
for(const k of order){const v=data[k];if(!v)continue;const sec=document.createElement('section');sec.id=k;const h=document.createElement('h2');h.textContent=labels[k]||k;sec.append(h);const link=document.createElement('a');link.href=k+'.json';link.textContent='机器可读登记';sec.append(link);let entries=[],metadata={};if(k==='next_experiment_gate'){entries=[{...v,name:'当前研究状态与下一步入口',_uid:k+':current:0'}]}else{for(const[n,x]of Object.entries(v)){if(Array.isArray(x)&&x.some(y=>y&&typeof y==='object'&&!Array.isArray(y))){x.forEach((y,i)=>{if(y&&typeof y==='object'&&!Array.isArray(y))entries.push({registry_group:n,...y,_uid:k+':'+n+':'+i})})}else metadata[n]=x}}if(Object.keys(metadata).length){const m=document.createElement('details'),q=document.createElement('summary');q.textContent='范围与解释';m.append(q);m.addEventListener('toggle',()=>{if(m.open&&!m.dataset.loaded){m.append(render(metadata));m.dataset.loaded='1'}});sec.append(m)}for(const e of entries){const d=document.createElement('details');d.id=e._uid;d.className='entry';d.dataset.registry=k;d.dataset.status=String(e.results_status||e.status||'');d.dataset.search=JSON.stringify(e).toLowerCase();const q=document.createElement('summary');q.textContent=(e.name||e.title||e.id||e.path||e.model||e.registry_group||'记录')+(e.results_status?' · '+e.results_status:'');d.append(q);d.addEventListener('toggle',()=>{if(!d.open||d.dataset.loaded)return;const body={...e};delete body._uid;d.append(render(body));const deps=catalog[e._uid]?.source_dependencies||[];if(deps.length){const box=document.createElement('details'),title=document.createElement('summary');title.textContent='来源文件进一步引用：显示 '+deps.length+' / '+catalog[e._uid].source_dependency_count+'（完整源链见 catalog.json）';box.append(title);box.addEventListener('toggle',()=>{if(box.open&&!box.dataset.loaded){box.append(render(deps));box.dataset.loaded='1'}});d.append(box)}const refs=catalog[e._uid]?.related_records||[];if(refs.length){const h=document.createElement('p');h.textContent='共享明确来源路径的关联卡片：显示 '+refs.length+' / '+catalog[e._uid].related_record_count+'（不自动表示因果支持；完整关联见 catalog.json）';d.append(h);const ul=document.createElement('ul'),seen=new Set();for(const r of refs){if(seen.has(r.uid))continue;seen.add(r.uid);const li=document.createElement('li'),a=document.createElement('a');a.href='#'+encodeURIComponent(r.uid);a.textContent=catalog[r.uid]?.title||r.uid;a.addEventListener('click',()=>{const target=document.getElementById(r.uid);if(target){target.classList.remove('hidden');target.open=true}});li.append(a);ul.append(li)}d.append(ul)}d.dataset.loaded='1'});sec.append(d)}document.getElementById('registries').append(sec);const option=document.createElement('option');option.value=k;option.textContent=labels[k];document.getElementById('registry-filter').append(option)}
for(const[k,v]of Object.entries(data.mechanisms.results_status_definitions||{})){const o=document.createElement('option');o.value=k;o.textContent=k;o.title=v;document.getElementById('status-filter').append(o)}
function filter(){const words=document.getElementById('search').value.toLowerCase().trim().split(/\s+/).filter(Boolean),registry=document.getElementById('registry-filter').value,status=document.getElementById('status-filter').value;let count=0;for(const e of document.querySelectorAll('.entry')){const yes=(!registry||registry===e.dataset.registry)&&(!status||(e.dataset.registry==='mechanisms'&&status===e.dataset.status))&&words.every(w=>e.dataset.search.includes(w));e.classList.toggle('hidden',!yes);if(yes)count++}document.getElementById('search-count').textContent='显示 '+count+' 张卡片'}
for(const id of ['search','registry-filter','status-filter'])document.getElementById(id).addEventListener(id==='search'?'input':'change',filter);filter();
function revealHash(){try{const e=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(e&&e.classList.contains('entry')){e.classList.remove('hidden');e.open=true}}catch{}}window.addEventListener('hashchange',revealHash);revealHash();
</script></body></html>'''
    overview = f"已索引 {inventory['total_indexed_entries']:,} 项资产、{inventory['run_directories']} 个运行目录；其中 {coverage['runs_with_curated_references']} 个运行目录有知识卡明确引用，其余 {len(coverage['runs_without_individual_curated_references'])} 个列入待关联队列。"
    page = page.replace("__COUNTS__", counts).replace("__TIME__", inventory["built_at_utc"]).replace("__DATA__", payload).replace("__LINKS__", link_payload).replace("__CATALOG__", catalog_payload).replace("__COVERAGE__", overview)
    (ARCHIVE / "index.html").write_text(page)


def build():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    data = curated_data()
    git = git_inventory()
    assets = []
    counts, sizes, runs = Counter(), Counter(), {}
    destination = ARCHIVE / "assets.jsonl"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for record in iter_assets():
            record["storage_class"] = storage_class(record["path"], git)
            assets.append(record)
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
    catalog, owners = make_catalog(data, assets, git)
    coverage = coverage_report(data, assets, runs, owners, git)
    inventory = {"schema_version": 2, "built_at_utc": datetime.now(timezone.utc).isoformat(),
                 "root": str(ROOT), "archive_mode": "in_place_index_original_assets_preserved",
                 "file_count_by_area": dict(counts), "bytes_by_area_no_symlink_target_sizes": dict(sizes),
                 "total_indexed_entries": sum(counts.values()), "run_directories": len(runs),
                 "areas": AREAS, "excluded_directory_names": sorted(SKIP_DIRS),
                 "self_archive_excluded_from_scan": True, "symlink_targets_not_traversed": True,
                 "environment_directories": environments,
                 "scope": "Metadata of local project assets, including vendor source. Files stay in place; external symlink targets are referenced. No claim that every old experiment is scientifically valid.",
                 "curated_registries": CURATED, "model_loads": 0, "environment_steps": 0, "api_requests": 0,
                 "checkpoint_hash_passes": 0,
                 "private_configuration_names_excluded": sorted(PRIVATE_NAMES),
                 "git_path_membership_not_remote_content_verification": True,
                 "knowledge_cards": catalog["entry_count"],
                 "pinned_submodules": git[1],
                 "curated_content_sha256": {name: hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() for name in CURATED},
                 "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "freshness_hash_scope": "Small registry JSON and this generator only; no checkpoint or bulk cache hashes."}
    dump(ARCHIVE / "runs.json", {"execution_metadata_only": True, "runs": list(runs.values())})
    dump(ARCHIVE / "inventory.json", inventory)
    dump(ARCHIVE / "catalog.json", catalog)
    dump(ARCHIVE / "coverage.json", coverage)
    write_index(data, inventory, catalog, coverage)
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


def search_records(query, registry, status, limit):
    catalog = json.loads((ARCHIVE / "catalog.json").read_text())
    words = query.casefold().split()
    matches = [entry for entry in catalog["entries"] if
               (not registry or entry["registry"] == registry) and
               (not status or entry["status"] == status) and
               all(word in entry["search_text"] for word in words)]
    for entry in matches[:limit]:
        print(json.dumps({key: entry[key] for key in ["uid", "registry", "group", "id", "title", "status", "local_references"]}, ensure_ascii=False))
    print(json.dumps({"matched": len(matches), "shown": min(len(matches), limit), "limit": limit}))


def check():
    data = curated_data()
    catalog = json.loads((ARCHIVE / "catalog.json").read_text())
    coverage = json.loads((ARCHIVE / "coverage.json").read_text())
    inventory = json.loads((ARCHIVE / "inventory.json").read_text())
    for name in CURATED:
        assert hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() == inventory.get("curated_content_sha256", {}).get(name), f"Stale registry index: {name}; run build"
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == inventory.get("generator_sha256"), "Generator changed; run build"
    page = (ARCHIVE / "index.html").read_text()
    embedded = re.search(r'<script id="archive-data" type="application/json">(.*?)</script>', page, re.S)
    assert embedded and json.loads(embedded.group(1)) == data, "HTML registry data is stale"
    links = re.search(r'<script id="archive-links" type="application/json">(.*?)</script>', page, re.S)
    assert links and all(not private_path(Path(path)) for path in json.loads(links.group(1))), "Private configuration in HTML links"
    uids = [entry["uid"] for entry in catalog["entries"]]
    assert len(uids) == len(set(uids)), "Duplicate knowledge card navigation IDs"
    uid_set = set(uids)
    for entry in catalog["entries"]:
        assert all(link["uid"] in uid_set for link in entry["related_records"])
    assets = [json.loads(line) for line in (ARCHIVE / "assets.jsonl").read_text().splitlines()]
    assert len(assets) == inventory["total_indexed_entries"] == coverage["total_indexed_assets"]
    assert all(not private_path(Path(item["path"])) for item in assets)
    registered = data["mechanisms"].get("config_to_entry", {})
    configs = [item["path"] for item in assets if item["area"] == "configs"]
    missing_configs = [path for path in configs if path not in registered]
    assert not missing_configs, f"Unmapped experiment configurations: {missing_configs}"
    print(json.dumps({"status": "passed", "indexed_assets": len(assets), "knowledge_cards": len(uids),
                      "mapped_experiment_configs": len(configs), "unresolved_path_like_mentions_for_review": len(coverage["missing_literal_path_mentions"]),
                      "scope": "Archive structure/coverage only; no scientific revalidation or model execution"}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    search = sub.add_parser("find"); search.add_argument("query"); search.add_argument("--area", choices=AREAS)
    search.add_argument("--limit", type=int, default=30)
    knowledge = sub.add_parser("search", help="Search scientific records, not only filenames")
    knowledge.add_argument("query")
    knowledge.add_argument("--registry", choices=[name.removesuffix(".json") for name in CURATED])
    knowledge.add_argument("--status")
    knowledge.add_argument("--limit", type=int, default=10)
    sub.add_parser("check", help="Validate archive structure and config coverage without running experiments")
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "check":
        check()
    else:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        if args.command == "find":
            find(args.query, args.area, args.limit)
        else:
            search_records(args.query, args.registry, args.status, args.limit)


if __name__ == "__main__":
    main()
