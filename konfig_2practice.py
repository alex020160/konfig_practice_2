# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import gzip
import io
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from collections import deque, defaultdict
from typing import Deque, Dict, Set, Tuple, Iterable, Callable
from urllib.request import url2pathname 



class ConfigError(Exception):
    def __init__(self, message: str, *, hint: Optional[str] = None):
        super().__init__(message)
        self.hint = hint


_RE_PKG_NAME = re.compile(r"^[A-Za-z0-9._\-]+$")

ALLOWED_MODES = {"readonly", "replay", "record", "mock"}

def validate_package_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        raise ConfigError("Имя пакета не задано.", hint='Укажите <package name="..."/>.')
    if not _RE_PKG_NAME.fullmatch(s):
        raise ConfigError(f"Недопустимое имя пакета: '{s}'", hint="Разрешены: буквы/цифры/._-")
    return s

def classify_repo_value(value: str) -> tuple[str, str]:
    """Определяем, это URL/файл/путь. На Этапе 2 используем только URL к Packages/Packages.gz"""
    raw = (value or "").strip()
    if not raw:
        raise ConfigError("Значение репозитория пустое.", hint="Укажите URL или путь.")
    parsed = urlparse(raw)

    if parsed.scheme in ("http", "https"):
        if not parsed.netloc:
            raise ConfigError("URL репозитория без хоста.", hint="Пример: https://archive.ubuntu.com/ubuntu/...")
        return ("url", raw)

    if parsed.scheme == "file":
        return ("fileurl", raw)

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return ("path", str(path))

def validate_mode(mode: str) -> str:
    s = (mode or "").strip().lower()
    if not s:
        raise ConfigError("Режим не задан.", hint=f"Допустимые: {', '.join(sorted(ALLOWED_MODES))}.")
    if s not in ALLOWED_MODES:
        raise ConfigError(f"Недопустимый режим: '{s}'", hint=f"Допустимые: {', '.join(sorted(ALLOWED_MODES))}.")
    return s

def normalize_filter(sub: Optional[str]) -> str:
    return "" if sub is None else sub.strip()


@dataclass
class AppConfig:
    package_name: str
    repo_kind: str      
    repo_value: str     
    test_repo_mode: str
    filter_substring: str

    @staticmethod
    def from_xml(config_path: str) -> "AppConfig":
        try:
            tree = ET.parse(config_path)
            root = tree.getroot()
        except FileNotFoundError:
            raise ConfigError(f"Файл конфигурации не найден: {config_path}", hint="Передайте верный путь в --config.")
        except ET.ParseError as e:
            raise ConfigError(f"Некорректный XML: {e}", hint="Проверьте теги/кавычки/кодировку.")
        except Exception as e:
            raise ConfigError(f"Ошибка чтения конфигурации: {e}")

        if root.tag not in ("config", "depviz"):
            raise ConfigError(f"Ожидался корневой тег <config> или <depviz>, получено <{root.tag}>.")

        pkg_el = root.find("package")
        if pkg_el is None or "name" not in pkg_el.attrib:
            raise ConfigError('Не найден элемент <package name="..."/>.')
        package_name = validate_package_name(pkg_el.attrib.get("name", ""))

        repo_el = root.find("repo")
        if repo_el is None or "value" not in repo_el.attrib:
            raise ConfigError('Не найден элемент <repo value="..."/>.', hint="Укажите http(s)://, file:// или путь.")
        repo_kind, repo_value = classify_repo_value(repo_el.attrib.get("value", ""))

        mode_el = root.find("mode")
        if mode_el is None or "value" not in mode_el.attrib:
            raise ConfigError('Не найден элемент <mode value="..."/>.', hint=f"Допустимые: {', '.join(sorted(ALLOWED_MODES))}.")
        test_repo_mode = validate_mode(mode_el.attrib.get("value", ""))

        filter_el = root.find("filter")
        filter_substring = normalize_filter(None if filter_el is None else filter_el.attrib.get("substring"))

        return AppConfig(
            package_name=package_name,
            repo_kind=repo_kind,
            repo_value=repo_value,
            test_repo_mode=test_repo_mode,
            filter_substring=filter_substring,
        )

#2 этап

def _is_packages_url(url: str) -> bool:
    return url.endswith("/Packages") or url.endswith("/Packages.gz") or url.endswith("/Packages.xz")

def _fetch_packages_text(url: str) -> str:

    req = Request(url, headers={"User-Agent": "depviz/0.1"})
    try:
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
    except HTTPError as e:
        raise ConfigError(f"HTTP ошибка при загрузке {url}: {e.code} {e.reason}")
    except URLError as e:
        raise ConfigError(f"Не удалось подключиться к {url}: {e.reason}")
    except Exception as e:
        raise ConfigError(f"Ошибка сети при загрузке {url}: {e}")

    if url.endswith(".gz"):
        try:
            data = gzip.decompress(data)
        except Exception as e:
            raise ConfigError(f"Не удалось распаковать gzip для {url}: {e}")
    elif url.endswith(".xz"):
        raise ConfigError("Формат .xz не поддержан на этом этапе.", hint="Используйте URL на Packages или Packages.gz")

    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")

def _parse_stanzas(packages_text: str) -> List[dict]:

    stanzas: List[dict] = []
    cur: dict = {}
    last_key: Optional[str] = None

    for line in packages_text.splitlines():
        if not line.strip():
            if cur:
                stanzas.append(cur)
                cur = {}
                last_key = None
            continue

        if line[0].isspace() and last_key:
            cur[last_key] += "\n" + line.strip()
            continue

        if ":" in line:
            key, val = line.split(":", 1)
            cur[key.strip()] = val.strip()
            last_key = key.strip()
        else:
            continue

    if cur:
        stanzas.append(cur)
    return stanzas

_DEP_ITEM_RE = re.compile(
        r"""
        ^\s*
        (?P<name>[A-Za-z0-9.+-]+)      # имя пакета
        (?:\s*:\s*[A-Za-z0-9-]+)?      # архитектура (например, :any)
        (?:\s*\([^)]+\))?              # версия в скобках
        \s*$
        """, re.X,
)

def _parse_depends(dep_str: str) -> list[list[str]]:
    """
    'A (>=1.0), B | C:any, D' -> [['A'], ['B', 'C'], ['D']]
    """
    result: list[list[str]] = []
    chunks = [c.strip() for c in dep_str.split(",") if c.strip()]
    for chunk in chunks:
        alts = [a.strip() for a in chunk.split("|") if a.strip()]
        names: list[str] = []
        for a in alts:
            m = _DEP_ITEM_RE.match(a)
            if m:
                names.append(m.group("name"))
        if names:
            result.append(names)
    return result

def get_direct_dependencies(packages_url: str, pkg_name: str) -> list[list[str]]:
        if not _is_packages_url(packages_url):
            raise ConfigError(
                "Ожидается URL на файл Packages или Packages.gz.",
                hint="Например: https://archive.ubuntu.com/ubuntu/dists/jammy-updates/main/binary-amd64/Packages.gz",
            )

        text = _fetch_packages_text(packages_url)
        stanzas = _parse_stanzas(text)

        candidates = [st for st in stanzas if st.get("Package") == pkg_name]
        if not candidates:
            raise ConfigError(
                f"Пакет '{pkg_name}' не найден в индексе Packages.",
                hint="Проверьте название, компонент (main/universe), карман (jammy, jammy-updates, jammy-security) и архитектуру."
            )

        chosen = None
        for st in candidates:
            if st.get("Depends", "").strip() or st.get("Pre-Depends", "").strip():
                chosen = st
                break
        if chosen is None:
            chosen = candidates[-1]  

        sys.stdout.write(
            f"[DEBUG] Нашли {len(candidates)} записей для '{pkg_name}'. "
            f"Выбрана версия: {chosen.get('Version','<нет>')}. "
            f"Depends: {'есть' if chosen.get('Depends','').strip() else 'нет'}, "
            f"Pre-Depends: {'есть' if chosen.get('Pre-Depends','').strip() else 'нет'}\n"
        )

        depends = chosen.get("Depends", "").strip()
        pre_depends = chosen.get("Pre-Depends", "").strip()

        merged = []
        if depends:
            merged.append(depends)
        if pre_depends:
            merged.append(pre_depends)

        if not merged:
            return []

    
        return _parse_depends(", ".join(merged))

#3 этап

_UPPER_NAME_RE = re.compile(r"^[A-Z]+$")

def load_test_repo_graph(path_or_fileurl: str) -> Dict[str, list[list[str]]]:
    p = urlparse(path_or_fileurl)
    if p.scheme == "file":
        path = Path(url2pathname(p.path))
    else:
        path = Path(path_or_fileurl)

    if not path.is_file():
        raise ConfigError(f"Файл тестового репозитория не найден: {path}")

    text = path.read_text(encoding="utf-8")

    index: Dict[str, List[List[str]]] = {}
    all_names: Set[str] = set()

    for i, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ConfigError(f"Строка {i}: ожидается 'NAME: deps...'", hint="Например: A: B C")
        left, right = [x.strip() for x in line.split(":", 1)]
        if not _UPPER_NAME_RE.fullmatch(left):
            raise ConfigError(f"Строка {i}: недопустимое имя '{left}'", hint="Имя должно быть A..Z")
        deps: list[str] = []
        if right:
            for name in right.split():
                if not _UPPER_NAME_RE.fullmatch(name):
                    raise ConfigError(f"Строка {i}: недопустимое имя завис-ти '{name}'", hint="Имя должно быть A..Z")
                deps.append(name)
        index[left] = [[d] for d in deps]
        all_names.add(left)
        all_names.update(deps)

    for n in list(all_names):
        index.setdefault(n, [])
    return index
#граф, BFS и операции 

def choose_from_alternatives(alts: list[str]) -> str:
    """Детерминированно выбираем одну альтернативу: лексикографический минимум."""
    return sorted(alts)[0]

def build_graph_bfs(
    get_deps: Callable[[str], list[list[str]]],
    root: str,
    *,
    ignore_sub: str = ""
) -> Tuple[Dict[str, list[str]], Dict[str, list[list[str]]]]:
    ignore = (ignore_sub or "").lower()
    def allowed(name: str) -> bool:
        return not (ignore and (ignore in name.lower()))

    adj: Dict[str, list[str]] = {}
    raw: Dict[str, list[list[str]]] = {}
    q: Deque[str] = deque()
    seen: Set[str] = set()

    if allowed(root):
        q.append(root)

    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)

        deps_alts = get_deps(u) or []
        raw[u] = deps_alts

        concrete: list[str] = []
        for alts in deps_alts:
            filtered = [x for x in alts if allowed(x)]
            if not filtered:
                continue
            v = choose_from_alternatives(filtered)
            if allowed(v):
                concrete.append(v)
        adj[u] = concrete

        for v in concrete:
            if v not in seen:
                q.append(v)

    adj.setdefault(root, adj.get(root, []))
    raw.setdefault(root, raw.get(root, []))
    return adj, raw

def reverse_graph(adj: Dict[str, list[str]]) -> Dict[str, Set[str]]:
    rg: Dict[str, Set[str]] = defaultdict(set)
    nodes: Set[str] = set(adj.keys())
    for vs in adj.values():
        nodes.update(vs)
    for u in nodes:
        rg.setdefault(u, set())
    for u, vs in adj.items():
        for v in vs:
            rg[v].add(u)
    return rg

def transitive_dependencies(adj: Dict[str, list[str]], start: str) -> Set[str]:
    seen: Set[str] = set()
    q: Deque[str] = deque(adj.get(start, []))
    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        for v in adj.get(u, []):
            if v not in seen:
                q.append(v)
    return seen

def dependents(adj: Dict[str, list[str]], target: str) -> Set[str]:
    rg = reverse_graph(adj)
    seen: Set[str] = set()
    q: Deque[str] = deque(rg.get(target, []))
    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        for v in rg.get(u, []):
            if v not in seen:
                q.append(v)
    return seen

def kahn_toposort_and_cycle_nodes(adj: Dict[str, list[str]]) -> Tuple[list[str], Set[str]]:
    nodes: Set[str] = set(adj.keys())
    for vs in adj.values():
        nodes.update(vs)
    indeg: Dict[str, int] = {u: 0 for u in nodes}
    for u, vs in adj.items():
        for v in vs:
            indeg[v] += 1
    q: Deque[str] = deque([u for u in nodes if indeg[u] == 0])
    topo: list[str] = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    cycle_nodes: Set[str] = {u for u, d in indeg.items() if d > 0}
    return topo, cycle_nodes

#вывод/визуализация

def print_edges(adj: Dict[str, list[str]]) -> None:
    print("\n[Этап 3] Рёбра графа (u -> v):")
    empty = True
    for u, vs in adj.items():
        for v in vs:
            print(f"{u} -> {v}")
            empty = False
    if empty:
        print("(нет рёбер)")

def print_bfs_levels(adj: Dict[str, list[str]], root: str) -> None:
    print("\n[Этап 3] Уровни BFS от корня:", root)
    seen: Set[str] = set([root])
    q: Deque[Tuple[str, int]] = deque([(root, 0)])
    levels: Dict[int, list[str]] = defaultdict(list)
    while q:
        u, d = q.popleft()
        levels[d].append(u)
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
    for d in sorted(levels):
        print(f"  [{d}] " + ", ".join(sorted(levels[d])))

def print_toposort_or_cycles(adj: Dict[str, list[str]]) -> None:
    topo, cycle_nodes = kahn_toposort_and_cycle_nodes(adj)
    if cycle_nodes:
        print("\n[Этап 3] Узлы, участвующие в циклах:")
        print(", ".join(sorted(cycle_nodes)))
    else:
        print("\n[Этап 3] Топологический порядок:")
        if topo:
            print(" -> ".join(topo))
        else:
            print("(граф пуст)")

def print_transitive_ops(adj: Dict[str, list[str]], root: str) -> None:
    deps = transitive_dependencies(adj, root)
    print("\n[Этап 3] Транзитивные зависимости корня:")
    print(", ".join(sorted(deps)) if deps else "(нет зависимостей)")

    rev = dependents(adj, root)
    print("\n[Этап 3] Пакеты, зависящие (транзитивно) от корня:")
    print(", ".join(sorted(rev)) if rev else "(нет зависимых)")

def print_dot(adj: Dict[str, list[str]], root: str) -> None:
    print("\n[Этап 3] DOT-граф:")
    print("digraph deps {")
    print("  rankdir=LR;")
    print(f'  "{root}" [shape=box, style=filled, fillcolor="#eef"];')
    nodes: Set[str] = set([root])
    for u, vs in adj.items():
        nodes.add(u)
        nodes.update(vs)
        for v in vs:
            print(f'  "{u}" -> "{v}";')
    for n in nodes:
        if n != root:
            print(f'  "{n}";')
    print("}")

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="depviz",
        description="Этап 1+2+3: XML-конфиг, прямые Depends, транзитивный граф BFS (с тестовым режимом)."
    )
    p.add_argument("--config", "-c", required=True, help="Путь к XML-конфигу.")
    return p.parse_args(argv)

def exclude_in_alternatives(dep_groups: list[list[str]], substring: str) -> list[list[str]]:
    if not substring:
        return dep_groups
    sub = substring.lower()
    cleaned: list[list[str]] = []
    for alts in dep_groups:
        kept = [name for name in alts if sub not in name.lower()]
        if kept:
            cleaned.append(kept)
    return cleaned

def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        cfg = AppConfig.from_xml(args.config)
    except ConfigError as e:
        sys.stderr.write(f"[ОШИБКА] {e}\n")
        if e.hint:
            sys.stderr.write(f"         Подсказка: {e.hint}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"[ОШИБКА] Неожиданная ошибка: {e}\n")
        return 3

    #1 этап
    print(f"package_name={cfg.package_name}")
    print(f"repo_kind={cfg.repo_kind}")
    print(f"repo_value={cfg.repo_value}")
    print(f"test_repo_mode={cfg.test_repo_mode}")
    print(f"filter_substring={cfg.filter_substring}")

    # 2 этап
    try:
        if cfg.repo_kind != "url":
            print("\n[Этап 2] Пропущено: repo_kind не 'url'. Для URL показываем только прямые Depends.")
        else:
            deps = get_direct_dependencies(cfg.repo_value, cfg.package_name)

            #изменение с этапом 3
            if cfg.filter_substring:
                deps = exclude_in_alternatives(deps, cfg.filter_substring)

            print("\n[Этап 2] Прямые зависимости (Depends) для пакета:", cfg.package_name)
            if not deps:
                print("(зависимостей не найдено после применения фильтра или поле Depends пустое)")
            else:
                for i, alts in enumerate(deps, 1):
                    print(f"{i}. " + (" | ".join(alts) if len(alts) > 1 else alts[0]))

    except ConfigError as e:
        sys.stderr.write(f"[ОШИБКА Этап 2] {e}\n")
        if e.hint:
            sys.stderr.write(f"               Подсказка: {e.hint}\n")
        return 4
    except Exception as e:
        sys.stderr.write(f"[ОШИБКА Этап 2] Неожиданная ошибка: {e}\n")
        return 5

    #3 этап
    try:
        ignore = cfg.filter_substring

        if cfg.repo_kind == "url":
            #репозиторий
            def _fetch(name: str) -> list[list[str]]:
                try:
                    return get_direct_dependencies(cfg.repo_value, name)
                except ConfigError:
                    #отсутствие узла - отсутствие зависимостей
                    return []
            adj, raw = build_graph_bfs(_fetch, cfg.package_name, ignore_sub=ignore)

        else:
            #тестовый режим
            test_index = load_test_repo_graph(cfg.repo_value)
            def _fetch(name: str) -> list[list[str]]:
                return test_index.get(name, [])
            adj, raw = build_graph_bfs(_fetch, cfg.package_name, ignore_sub=ignore)

        print_edges(adj)
        print_toposort_or_cycles(adj)          # либо топопорядок, либо узлы циклов
        print_bfs_levels(adj, cfg.package_name)
        print_transitive_ops(adj, cfg.package_name)
        print_dot(adj, cfg.package_name)

    except ConfigError as e:
        sys.stderr.write(f"[ОШИБКА Этап 3] {e}\n")
        if e.hint:
            sys.stderr.write(f"               Подсказка: {e.hint}\n")
        return 6
    except Exception as e:
        sys.stderr.write(f"[ОШИБКА Этап 3] Неожиданная ошибка: {e}\n")
        return 7

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

