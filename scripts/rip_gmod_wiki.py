from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

BASE = "https://wiki.facepunch.com"
ROOT_PAGE = "/gmod/"
OUT_PATH = Path("full_gmod_api.json")
HEADERS = {"User-Agent": "py2glua-scraper/0.1"}
session = requests.Session()
session.headers.update(HEADERS)


# Utility
def fetch_soup(url: str) -> BeautifulSoup:
    r = session.get(url, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def text_clean(tag: Tag | None) -> str | None:
    if not tag:
        return None

    return tag.get_text(" ", strip=True) or None


def clean_title(text: str) -> str:
    text = re.sub(r"\s*\d+\s*$", "", text)
    return text.strip()


def extract_realm(classes: set[str]) -> list[str]:
    has_client = "rc" in classes
    has_server = "rs" in classes
    has_menu = "rm" in classes

    realms = []

    if has_client and has_server:
        realms.append("SHARED")

    else:
        if has_client:
            realms.append("CLIENT")

        if has_server:
            realms.append("SERVER")

    if has_menu:
        realms.append("MENU")

    return realms


def extract_kind(classes: set[str]) -> str:
    if "meth" in classes:
        return "method"

    if "field" in classes:
        return "field"

    if "class" in classes:
        return "class"

    if "enum" in classes:
        return "enum"

    return "unknown"


# Developer Reference parsing
def find_dev_reference_section(soup: BeautifulSoup) -> Tag:
    for div in soup.find_all("div", class_="sectionheader"):
        if div.get_text(strip=True) == "Developer Reference":
            section = div.find_next_sibling("div", class_="section")
            if not section:
                raise RuntimeError("Developer Reference section container not found")

            return section

    raise RuntimeError("Developer Reference header not found")


def parse_developer_reference() -> dict[str, dict[str, list[dict]]]:
    soup = fetch_soup(urljoin(BASE, ROOT_PAGE))
    section = find_dev_reference_section(soup)

    result: dict[str, dict[str, list[dict]]] = {}

    for details in section.find_all("details", class_="level1", recursive=False):
        raw_title = details.find("summary").get_text(strip=True)  # type: ignore
        category = clean_title(raw_title)

        if category == "Miscellaneous Reference":
            continue

        result[category] = {}

        for a in details.find_all("a", href=True):
            classes = set(a.get("class", []))  # type: ignore

            if "depr" in classes or "intrn" in classes:
                continue

            href = a["href"].strip()  # type: ignore
            if not href.startswith("/gmod/"):
                continue

            kind = extract_kind(classes)
            realms = extract_realm(classes)

            result[category].setdefault(kind, []).append(
                {
                    "title": a.get_text(strip=True),
                    "url": urljoin(BASE, href),
                    "realms": realms,
                }
            )

    return result


# Function page parsing
def parse_description(block: Tag) -> list[dict]:
    desc = block.find("div", class_="function_description")
    if not desc:
        return []

    result: list[dict] = []

    for child in desc.find_all(recursive=False):
        classes = set(child.get("class", []))  # type: ignore
        text = child.get_text(" ", strip=True)
        if not text:
            continue

        if "note" in classes:
            result.append({"type": "note", "text": text})

        elif "warning" in classes:
            result.append({"type": "warning", "text": text})

        elif "bug" in classes:
            result.append({"type": "bug", "text": text})

        elif child.name == "p":
            result.append({"type": "paragraph", "text": text})

        else:
            result.append(
                {
                    "type": "unknown",
                    "tag": child.name,
                    "classes": list(classes),
                    "text": text,
                }
            )

    return result


def parse_arguments(block: Tag) -> list[dict]:
    args_section = block.find("div", class_="function_arguments")
    if not args_section:
        return []

    arguments: list[dict] = []

    for arg_div in args_section.find_all("div", recursive=False):
        type_tag = arg_div.find("a", class_="link-page")
        name_tag = arg_div.find("span", class_="name")
        default_tag = arg_div.find("span", class_="default")
        desc_div = arg_div.find("div", class_="numbertagindent")

        default = None
        if default_tag:
            default = default_tag.get_text(" ", strip=True)
            default = default.replace("= ", "", 1).strip()

        arguments.append(
            {
                "type": text_clean(type_tag),
                "name": text_clean(name_tag),
                "default": default,
                "description": text_clean(desc_div),
            }
        )

    return arguments


def parse_returns(block: Tag) -> list[dict]:
    ret_section = block.find("div", class_="function_returns")
    if not ret_section:
        return []

    returns: list[dict] = []

    for ret_div in ret_section.find_all("div", recursive=False):
        type_tag = ret_div.find("a", class_="link-page")
        name_tag = ret_div.find("span", class_="name")
        desc_div = ret_div.find("div", class_="numbertagindent")

        returns.append(
            {
                "type": text_clean(type_tag),
                "name": text_clean(name_tag),
                "description": text_clean(desc_div),
            }
        )

    return returns


def parse_function_page(url: str) -> dict:
    soup = fetch_soup(url)

    block = soup.find("div", class_="libraryfunc") or soup.find(
        "div", class_="function"
    )
    if not block:
        return {}

    line = block.find("div", class_="function_line")
    raw = line.get_text(" ", strip=True) if line else ""
    name = raw.split("(", 1)[0].strip()

    return {
        "name": name,
        "description": parse_description(block),
        "arguments": parse_arguments(block),
        "returns": parse_returns(block),
    }


# Main
def main():
    dev_ref = parse_developer_reference()

    full_output: dict = {}

    tasks = []
    task_map = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        for category, kinds in dev_ref.items():
            full_output[category] = {}

            for kind, items in kinds.items():
                full_output[category][kind] = []

                for item in items:
                    future = executor.submit(parse_function_page, item["url"])
                    tasks.append(future)
                    task_map[future] = (category, kind, item)

        for future in as_completed(tasks):
            category, kind, item = task_map[future]

            try:
                details = future.result()
            except Exception as e:
                print(f"Failed: {item['title']} -> {e}")
                details = {}

            full_output[category][kind].append(
                {
                    "title": item["title"],
                    "realms": item["realms"],
                    "details": details,
                }
            )

            print(f"Done: {item['title']}")

    OUT_PATH.write_text(
        json.dumps(full_output, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
