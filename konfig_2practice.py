import argparse
import gzip
import io
import os
import re
import sys
import textwrap
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import deque, defaultdict

# -----------------------------
# Ошибки/исключения
# -----------------------------

class ConfigError(Exception):
    pass

class RepositoryError(Exception):
    pass

class PackageNotFoundError(Exception):
    pass

# -----------------------------
# Парсер конфигурации (XML)
# -----------------------------

class AppConfig:
    def __init__(self, package_name: str, repo_url: str, mode: str, filter_substring: str):
        self.package_name = package_name
        self.repo_url = repo_url
        self.mode = mode
        self.filter_substring = filter_substring

    @staticmethod
    def from_xml(path: str) -> "AppConfig":
        if not os.path.exists(path):
            raise ConfigError(f"Файл конфигурации не найден: {path}")

        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ConfigError(f"Некорректный XML: {e}")

        def get_text(tag: str, required=True, allow_empty=False) -> str:
            el = root.find(tag)
            if el is None:
                if required:
                    raise ConfigError(f"Отсутствует обязательный тег <{tag}>")
                return ""
            val = (el.text or "").strip()
            if not allow_empty and val == "":
                raise ConfigError(f"Тег <{tag}> пустой")
            return val

        package_name = get_text("package_name")
        repo_url = get_text("repo_url")
        mode = get_text("mode")
        filter_substring = get_text("filter_substring", required=False, allow_empty=True)

        mode = mode.lower()
        if mode not in ("real", "test"):
            raise ConfigError("Тег <mode> должен быть 'real' или 'test'")

        if mode == "real":
            # Валидируем URL (разрешаем http/https и прямую ссылку на Packages/Packages.gz)
            parsed = urllib.parse.urlparse(repo_url)
            if parsed.scheme not in ("http", "https"):
                raise ConfigError("В режиме 'real' repo_url должен быть http/https URL на Packages или Packages.gz")
            if not (parsed.path.endswith("Packages") or parsed.path.endswith("Packages.gz")):
                raise ConfigError("В режиме 'real' repo_url должен указывать прямо на Packages или Packages.gz")
        else:
            # test — должен быть существующим файлом
            if not os.path.exists(repo_url):
                raise ConfigError(f"В режиме 'test' файл репозитория не найден: {repo_url}")

        return AppConfig(package_name, repo_url, mode, filter_substring)

    def as_kv(self) -> dict:
        return {
            "package_name": self.package_name,
            "repo_url": self.repo_url,
            "mode": self.mode,
            "filter_substring": self.filter_substring
        }

# -----------------------------
# Разбор репозитория APT (Packages/Packages.gz) и тестового файла
# -----------------------------

APT_STANZA_SEP_RE = re.compile(r"\n\s*\n", re.MULTILINE)

def _http_get(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read()
    except Exception as e:
        raise RepositoryError(f"Не удалось скачать {url}: {e}")

def _read_packages_blob(repo_url: str) -> str:
    blob = _http_get(repo_url)
    # Если gz — распаковываем
    if repo_url.endswith(".gz"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(blob)) as gz:
                data = gz.read()
        except OSError as e:
            raise RepositoryError(f"Не удалось распаковать Packages.gz: {e}")
        return data.decode("utf-8", errors="replace")
    else:
        return blob.decode("utf-8", errors="replace")

def _parse_apt_packages(packages_text: str) -> dict:
    """
    Возвращает dict: name -> dict(fields)
    Поля интереса: Package, Depends, Pre-Depends
    """
    result = {}
    stanzas = APT_STANZA_SEP_RE.split(packages_text.strip() + "\n\n")
    for stanza in stanzas:
        if not stanza.strip():
            continue
        fields = {}
        # Debian control-подобный формат: ключ: значение (+ возможно continuation lines)
        key = None
        for line in stanza.splitlines():
            if not line:
                continue
            if re.match(r"^\S+:", line):
                k, v = line.split(":", 1)
                key = k.strip()
                fields[key] = v.strip()
            else:
                # продолжение предыдущего поля
                if key:
                    fields[key] += " " + line.strip()
        pkg = fields.get("Package")
        if pkg:
            result[pkg] = fields
    return result

# Разбор Depends/Pre-Depends:
# - делим по ',' на группы
# - внутри группы делим по '|' на альтернативы, берём первую
# - чистим имя: обрезаем версии в скобках и суффиксы архитектур ':any', ':amd64' etc.
NAME_CLEAN_RE = re.compile(r"^[a-z0-9+.-]+", re.IGNORECASE)

def _split_deps_field(value: str) -> list[str]:
    if not value:
        return []
    groups = [g.strip() for g in value.split(",") if g.strip()]
    deps = []
    for g in groups:
        alt = [x.strip() for x in g.split("|") if x.strip()]
        if not alt:
            continue
        first = alt[0]
        # удаляем версии в скобках
        first = re.sub(r"\(.*?\)", "", first).strip()
        # удаляем архитектурные суффиксы
        first = re.sub(r":[a-z0-9]+$", "", first, flags=re.IGNORECASE).strip()
        m = NAME_CLEAN_RE.match(first)
        if m:
            deps.append(m.group(0))
    return deps

def load_real_repo(repo_url: str) -> dict[str, list[str]]:
    """
    Возвращает отображение: package -> direct_deps
    """
    text = _read_packages_blob(repo_url)
    table = _parse_apt_packages(text)
    graph = {}
    for name, fields in table.items():
        deps = []
        deps += _split_deps_field(fields.get("Pre-Depends", ""))
        deps += _split_deps_field(fields.get("Depends", ""))
        # уникализируем, сохраняя порядок
        seen = set()
        dedup = []
        for d in deps:
            if d not in seen:
                seen.add(d)
                dedup.append(d)
        graph[name] = dedup
    return graph

def load_test_repo(path: str) -> dict[str, list[str]]:
    """
    Формат строк: "A: B C" или "C:".
    Возвращает: package -> direct_deps
    """
    if not os.path.exists(path):
        raise RepositoryError(f"Файл не найден: {path}")
    graph = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise RepositoryError(f"Строка {i}: отсутствует ':'")
            left, right = line.split(":", 1)
            pkg = left.strip()
            if not re.fullmatch(r"[A-Z]+", pkg):
                raise RepositoryError(f"Строка {i}: имя пакета должно быть из заглавных латинских букв")
            deps = [x for x in right.strip().split() if x]
            for d in deps:
                if not re.fullmatch(r"[A-Z]+", d):
                    raise RepositoryError(f"Строка {i}: имя зависимости '{d}' некорректно")
            graph[pkg] = deps
    # гарантируем наличие пустых списков для одиноких узлов
    for deps in list(graph.values()):
        for d in deps:
            graph.setdefault(d, [])
    return graph

# -----------------------------
# Построение графа (BFS без рекурсии) + фильтр + циклы
# -----------------------------

def build_graph_bfs(repo_graph: dict[str, list[str]],
                    root: str,
                    exclude_substring: str) -> dict[str, set[str]]:
    """
    Возвращает подграф, достижимый из root, учитывая фильтр по подстроке.
    Формат: name -> set(children)
    """
    excl = (exclude_substring or "").lower()
    def is_excluded(name: str) -> bool:
        return excl != "" and excl in name.lower()

    if root not in repo_graph:
        raise PackageNotFoundError(f"Пакет '{root}' не найден в репозитории")

    result: dict[str, set[str]] = defaultdict(set)
    visited: set[str] = set()
    q = deque()

    if not is_excluded(root):
        q.append(root)
        visited.add(root)
    else:
        # Корневой пакет отфильтрован — граф будет пустым, но это корректно
        return {}

    while q:
        cur = q.popleft()
        for dep in repo_graph.get(cur, []):
            if is_excluded(dep):
                continue
            result[cur].add(dep)
            if dep not in visited:
                visited.add(dep)
                q.append(dep)
            # если dep уже посещён — просто не добавляем в очередь (цикл обработан)
    # убедимся, что все вершины присутствуют как ключи
    for u, childs in list(result.items()):
        for v in childs:
            result.setdefault(v, set())
    return result

# -----------------------------
# Этап 4: Порядок загрузки (топологическая сортировка)
# -----------------------------

def install_order_kahn(graph: dict[str, set[str]]) -> tuple[list[str], set[tuple[str, str]]]:
    """
    graph: u -> {v1, v2, ...}  (u зависит от v)
    Возвращает (порядок, множество рёбер, остающихся в цикле)
    """
    indeg = defaultdict(int)
    for u in graph:
        indeg.setdefault(u, 0)
    for u, vs in graph.items():
        for v in vs:
            indeg[v] += 1

    q = deque([u for u, d in indeg.items() if d == 0])
    order = []
    # копия графа
    g = {u: set(vs) for u, vs in graph.items()}

    while q:
        u = q.popleft()
        order.append(u)
        # «удаляем» вершину u и рёбра u->v
        for v in list(g.get(u, [])):
            g[u].remove(v)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # рёбра, которые остались — часть циклов
    remaining_edges = set()
    for u, vs in g.items():
        for v in vs:
            remaining_edges.add((u, v))
    return order, remaining_edges

# -----------------------------
# Этап 5: Визуализация в Graphviz DOT
# -----------------------------

def to_dot(graph: dict[str, set[str]], root: str) -> str:
    lines = ["digraph deps {", '  rankdir=LR;']
    # выделим корень
    if root in graph:
        lines.append(f'  "{root}" [shape=doublecircle];')
    # узлы/рёбра
    nodes = set(graph.keys())
    for vs in graph.values():
        nodes.update(vs)
    for n in sorted(nodes):
        if n == root:
            continue
        lines.append(f'  "{n}" [shape=circle];')
    for u, vs in graph.items():
        for v in vs:
            lines.append(f'  "{u}" -> "{v}";')
    lines.append("}")
    return "\n".join(lines)

# -----------------------------
# Печать «прямых» зависимостей
# -----------------------------

def direct_deps_of(repo_graph: dict[str, list[str]],
                   package: str,
                   exclude_substring: str) -> list[str]:
    excl = (exclude_substring or "").lower()
    if package not in repo_graph:
        raise PackageNotFoundError(f"Пакет '{package}' не найден в репозитории")
    deps = []
    for d in repo_graph.get(package, []):
        if excl and excl in d.lower():
            continue
        deps.append(d)
    return deps

# -----------------------------
# CLI
# -----------------------------

def stage1_print_config(cfg: AppConfig):
    print("Параметры конфигурации (ключ=значение):")
    for k, v in cfg.as_kv().items():
        print(f"{k}={v}")

def stage2_print_direct(cfg: AppConfig, repo_graph: dict[str, list[str]]):
    deps = direct_deps_of(repo_graph, cfg.package_name, cfg.filter_substring)
    print(f"Прямые зависимости пакета '{cfg.package_name}':")
    if deps:
        for d in deps:
            print(f"  - {d}")
    else:
        print("  (нет)")

def stage3_build_graph(cfg: AppConfig, repo_graph: dict[str, list[str]]):
    subgraph = build_graph_bfs(repo_graph, cfg.package_name, cfg.filter_substring)
    print(f"Построен граф зависимостей (BFS) для '{cfg.package_name}':")
    print(f"  узлов: {len(subgraph) or 0}")
    edges = sum(len(vs) for vs in subgraph.values())
    print(f"  рёбер: {edges}")
    # Небольшая сводка
    for u in sorted(subgraph.keys()):
        vs = ", ".join(sorted(subgraph[u])) if subgraph[u] else "—"
        print(f"  {u} -> {vs}")

def stage4_install_order(cfg: AppConfig, repo_graph: dict[str, list[str]]):
    subgraph = build_graph_bfs(repo_graph, cfg.package_name, cfg.filter_substring)
    order, remains = install_order_kahn(subgraph)
    print(f"Порядок загрузки (топологическая сортировка) для '{cfg.package_name}':")
    if order:
        print("  " + " -> ".join(order))
    else:
        print("  (пусто)")
    if remains:
        print("Внимание: обнаружены циклы, рёбра внутри цикла(ов):")
        for u, v in sorted(remains):
            print(f"  {u} -> {v}")

    # Пояснение возможных расхождений с apt
    print("\nПримечание о возможных расхождениях с реальным менеджером пакетов:")
    print("- альтернативные зависимости (A | B) — здесь берётся первая альтернатива; apt может выбрать другую;")
    print("- Pre-Depends/скрипты postinst/Triggers — apt учитывает порядок иначе;")
    print("- виртуальные пакеты/Provides — в этой реализации не разрешаются;")
    print("- архитектуры и pin-приоритеты — игнорируются.")

def stage5_print_dot(cfg: AppConfig, repo_graph: dict[str, list[str]]):
    subgraph = build_graph_bfs(repo_graph, cfg.package_name, cfg.filter_substring)
    dot = to_dot(subgraph, cfg.package_name)
    print(dot)

def load_repo(cfg: AppConfig) -> dict[str, list[str]]:
    if cfg.mode == "real":
        return load_real_repo(cfg.repo_url)
    else:
        return load_test_repo(cfg.repo_url)

def main():
    parser = argparse.ArgumentParser(
        description="Визуализация графа зависимостей apt-пакетов (без менеджеров пакетов)."
    )
    parser.add_argument("--config", required=True, help="Путь к XML-файлу конфигурации")
    parser.add_argument("--stage", required=True, type=int, choices=[1, 2, 3, 4, 5],
                        help="Номер этапа для демонстрации")
    args = parser.parse_args()

    try:
        cfg = AppConfig.from_xml(args.config)
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    if args.stage == 1:
        # Только печать параметров и демонстрация валидации
        try:
            stage1_print_config(cfg)
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Для этапов 2–5 уже нужно загрузить репозиторий
    try:
        repo_graph = load_repo(cfg)
    except (RepositoryError, ConfigError) as e:
        print(f"[REPO ERROR] {e}", file=sys.stderr)
        sys.exit(3)

    try:
        if args.stage == 2:
            stage2_print_direct(cfg, repo_graph)
        elif args.stage == 3:
            stage3_build_graph(cfg, repo_graph)
        elif args.stage == 4:
            stage4_install_order(cfg, repo_graph)
        elif args.stage == 5:
            stage5_print_dot(cfg, repo_graph)
    except PackageNotFoundError as e:
        print(f"[PACKAGE ERROR] {e}", file=sys.stderr)
        sys.exit(4)
    except KeyboardInterrupt:
        print("Остановка по Ctrl+C", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[UNEXPECTED ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

