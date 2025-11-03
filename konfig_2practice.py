#!/usr/bin/env python3
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

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="depviz",
        description="Этап 1+2: XML-конфиг + вывод параметров + извлечение прямых зависимостей из APT Packages."
    )
    p.add_argument("--config", "-c", required=True, help="Путь к XML-конфигу.")
    return p.parse_args(argv)

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

    #Этап 1
    print(f"package_name={cfg.package_name}")
    print(f"repo_kind={cfg.repo_kind}")
    print(f"repo_value={cfg.repo_value}")
    print(f"test_repo_mode={cfg.test_repo_mode}")
    print(f"filter_substring={cfg.filter_substring}")

    # этап 2
    try:
        if cfg.repo_kind != "url":
            print("\n[Этап 2] Пропущено: repo_kind не 'url'. Укажите прямой URL на Packages/Packages.gz.")
            return 0

        deps = get_direct_dependencies(cfg.repo_value, cfg.package_name)

        if cfg.filter_substring:
            sub = cfg.filter_substring.lower()
            deps = [alts for alts in deps if any(sub in name.lower() for name in alts)]

        print("\n[Этап 2] Прямые зависимости (Depends) для пакета:", cfg.package_name)
        if not deps:
            print("(зависимостей не найдено или поле Depends пустое)")
        else:
            for i, alts in enumerate(deps, 1):
                if len(alts) == 1:
                    print(f"{i}. {alts[0]}")
                else:
                    print(f"{i}. " + " | ".join(alts))

    except ConfigError as e:
        sys.stderr.write(f"[ОШИБКА Этап 2] {e}\n")
        if e.hint:
            sys.stderr.write(f"               Подсказка: {e.hint}\n")
        return 4
    except Exception as e:
        sys.stderr.write(f"[ОШИБКА Этап 2] Неожиданная ошибка: {e}\n")
        return 5

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

