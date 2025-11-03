#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse
import xml.etree.ElementTree as ET



class ConfigError(Exception):
    """Базовая ошибка конфигурации с полем 'hint' для более удобного вывода."""
    def __init__(self, message: str, *, hint: Optional[str] = None):
        super().__init__(message)
        self.hint = hint



_RE_PKG_NAME = re.compile(r"^[A-Za-z0-9._\-]+$")

ALLOWED_MODES = {
    # - readonly: читаем только локальный файл (никаких изменений)
    # - replay: используем только то, что уже есть в файле, падать если не хватает данных
    # - record: в будущем будем дозаписывать в файл (сейчас просто валидное значение)
    # - mock: в будущем генерировать синтетические данные (сейчас просто валидное значение)
    "readonly",
    "replay",
    "record",
    "mock",
}


def _is_http_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https")
    except Exception:
        return False


def _is_file_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme == "file" and bool(p.path)
    except Exception:
        return False


def _classify_repo_target(value: str) -> Tuple[str, str]:
    raw = value.strip()
    if not raw:
        raise ConfigError("Пустое значение репозитория.", hint="Укажите URL или путь к файлу.")

    if _is_http_url(raw):
        return ("url", raw)

    if _is_file_url(raw):
        path = Path(urlparse(raw).path)
        if not path.is_file():
            raise ConfigError(
                f"Файл репозитория не найден: {path}",
                hint="Проверьте, что файл существует и доступен."
            )
        return ("file", str(path.resolve()))

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.is_file():
        raise ConfigError(
            f"Файл репозитория не найден: {path}",
            hint="Укажите существующий файл, либо используйте http(s):// или file:// URL."
        )
    return ("file", str(path))


def _validate_package_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        raise ConfigError("Имя пакета не задано.", hint="Задайте <package name=\"...\"/>.")
    if not _RE_PKG_NAME.fullmatch(s):
        raise ConfigError(
            f"Недопустимое имя пакета: '{s}'",
            hint="Разрешены латинские буквы, цифры, '.', '_' и '-'."
        )
    return s


def _validate_mode(mode: str) -> str:
    s = (mode or "").strip().lower()
    if not s:
        raise ConfigError("Режим не задан.", hint=f"Допустимые значения: {', '.join(sorted(ALLOWED_MODES))}.")
    if s not in ALLOWED_MODES:
        raise ConfigError(
            f"Недопустимый режим: '{s}'",
            hint=f"Допустимые значения: {', '.join(sorted(ALLOWED_MODES))}."
        )
    return s


def _validate_filter_substring(sub: Optional[str]) -> str:
    if sub is None:
        return ""
    s = sub.strip()
    if not s and sub != "":
        raise ConfigError("Подстрока фильтра содержит только пробельные символы.", hint="Либо укажите непустую подстроку, либо удалите элемент <filter/>.")
    return s


@dataclass
class AppConfig:
    package_name: str
    repo_kind: str   
    repo_value: str   
    test_repo_mode: str
    filter_substring: str

    @staticmethod
    def from_xml(path: Path) -> "AppConfig":
        try:
            tree = ET.parse(str(path))
            root = tree.getroot()
        except ET.ParseError as e:
            raise ConfigError(f"Некорректный XML: {e}", hint="Проверьте закрывающие теги и кавычки.")
        except FileNotFoundError:
            raise ConfigError(f"Файл конфигурации не найден: {path}", hint="Передайте правильный путь к --config.")
        except Exception as e:
            raise ConfigError(f"Ошибка чтения конфигурации: {e}")

        if root.tag != "depviz":
            raise ConfigError(
                f"Ожидался корневой тег <depviz>, получено <{root.tag}>.",
                hint="Обновите корневой тег на <depviz>."
            )

        pkg_el = root.find("package")
        if pkg_el is None or "name" not in pkg_el.attrib:
            raise ConfigError("Не найден элемент <package name=\"...\"/>.")
        package_name = _validate_package_name(pkg_el.attrib.get("name", ""))

        repo_el = root.find("repo")
        if repo_el is None or "value" not in repo_el.attrib:
            raise ConfigError("Не найден элемент <repo value=\"...\"/>.",
                              hint="Укажите URL (http/https/file) или путь к файлу JSON.")
        repo_kind, repo_value = _classify_repo_target(repo_el.attrib.get("value", ""))

        mode_el = root.find("mode")
        if mode_el is None or "value" not in mode_el.attrib:
            raise ConfigError("Не найден элемент <mode value=\"...\"/>.",
                              hint=f"Допустимые значения: {', '.join(sorted(ALLOWED_MODES))}.")
        test_repo_mode = _validate_mode(mode_el.attrib.get("value", ""))

        filter_el = root.find("filter")
        filter_substring = _validate_filter_substring(
            None if filter_el is None else filter_el.attrib.get("substring")
        )

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
        description="Этап 1: загрузка конфигурации из XML и вывод параметров."
    )
    p.add_argument(
        "--config", "-c",
        required=True,
        help="Путь к XML-конфигу (например, config.xml)."
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        cfg = AppConfig.from_xml(Path(args.config))
    except ConfigError as e:
        sys.stderr.write(f"[ОШИБКА] {e}\n")
        if getattr(e, "hint", None):
            sys.stderr.write(f"         Подсказка: {e.hint}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"[ОШИБКА] Неожиданная ошибка: {e}\n")
        return 3

    # Этап 1
    lines = [
        ("package_name", cfg.package_name),
        ("repo_kind", cfg.repo_kind),
        ("repo_value", cfg.repo_value),
        ("test_repo_mode", cfg.test_repo_mode),
        ("filter_substring", cfg.filter_substring),
    ]
    for k, v in lines:
        print(f"{k}={v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
