#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


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
        raise ConfigError(
            f"Недопустимое имя пакета: '{s}'",
            hint="Разрешены: буквы/цифры/._-"
        )
    return s

def classify_repo_value(value: str) -> tuple[str, str]:
    raw = (value or "").strip()
    if not raw:
        raise ConfigError("Значение репозитория пустое.", hint="Укажите URL или путь.")
    parsed = urlparse(raw)

    if parsed.scheme in ("http", "https"):
        if not parsed.netloc:
            raise ConfigError("URL репозитория без хоста.", hint="Пример: https://example.org/repo.json")
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
        raise ConfigError(
            f"Недопустимый режим: '{s}'",
            hint=f"Допустимые: {', '.join(sorted(ALLOWED_MODES))}."
        )
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
            raise ConfigError(f"Файл конфигурации не найден: {config_path}",
                              hint="Передайте верный путь в --config.")
        except ET.ParseError as e:
            raise ConfigError(f"Некорректный XML: {e}",
                              hint="Проверьте теги/кавычки/кодировку.")
        except Exception as e:
            raise ConfigError(f"Ошибка чтения конфигурации: {e}")
        if root.tag not in ("config", "depviz"):
            raise ConfigError(
                f"Ожидался корневой тег <config> или <depviz>, получено <{root.tag}>."
            )

        pkg_el = root.find("package")
        if pkg_el is None or "name" not in pkg_el.attrib:
            raise ConfigError('Не найден элемент <package name="..."/>.')
        package_name = validate_package_name(pkg_el.attrib.get("name", ""))

        repo_el = root.find("repo")
        if repo_el is None or "value" not in repo_el.attrib:
            raise ConfigError('Не найден элемент <repo value="..."/>.',
                              hint="Укажите http(s)://, file:// или путь.")
        repo_kind, repo_value = classify_repo_value(repo_el.attrib.get("value", ""))

        mode_el = root.find("mode")
        if mode_el is None or "value" not in mode_el.attrib:
            raise ConfigError('Не найден элемент <mode value="..."/>.',
                              hint=f"Допустимые: {', '.join(sorted(ALLOWED_MODES))}.")
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="depviz",
        description="Этап 1: разбор XML-конфига и печать параметров (без работы с репозиторием)."
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

    #этап1
    print(f"package_name={cfg.package_name}")
    print(f"repo_kind={cfg.repo_kind}")
    print(f"repo_value={cfg.repo_value}")
    print(f"test_repo_mode={cfg.test_repo_mode}")
    print(f"filter_substring={cfg.filter_substring}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
