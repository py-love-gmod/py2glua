from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup, Tag

BASE = "https://wiki.facepunch.com"
ROOT = "https://wiki.facepunch.com/gmod/"

OUT = Path("scripts/full_gmod_api.json")

HEADERS = {"User-Agent": "py2glua-scraper/0.1"}

DEV_MODE = False
DEV_SAMPLE = 1

CONCURRENT_REQUESTS = 32

SKIP_CATEGORIES = {
    "Shaders",
    "Miscellaneous Reference",
}

CLASSLIKE_CATEGORIES = {
    "Classes",
    "Libraries",
    "Panels",
    "Hooks",
}


async def fetch_soup(session: aiohttp.ClientSession, url: str) -> BeautifulSoup:
    last_exc: Exception | None = None

    for attempt in range(3):
        try:
            async with session.get(url) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status} for {url}: {text[:200]}")
                text = await resp.text()
                return BeautifulSoup(text, "lxml")
        except Exception as e:
            last_exc = e
            await asyncio.sleep(0.3 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch {url}") from last_exc


def text_clean(tag: Tag | None) -> str | None:
    if not tag:
        return None
    return tag.get_text(" ", strip=True) or None


def maybe_sample(items: list[dict]) -> list[dict]:
    return items[:DEV_SAMPLE] if DEV_MODE else items


def extract_realm(classes: set[str]) -> list[str]:
    has_client = "rc" in classes
    has_server = "rs" in classes
    has_menu = "rm" in classes

    realms: list[str] = []

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


def extract_type(classes: set[str]) -> str:
    if "meth" in classes:
        return "method"
    if "field" in classes:
        return "field"
    if "hook" in classes:
        return "hook"
    if "panel" in classes:
        return "panel"
    if "lib" in classes:
        return "library"
    if "gameevent" in classes:
        return "gameevent"
    if "enum" in classes:
        return "enum"
    if "type" in classes:
        return "class"
    return "unknown"


def is_skipped_link_classes(classes: set[str]) -> bool:
    return bool({"intrn", "depr"} & classes)


def find_dev_section(soup: BeautifulSoup) -> Tag:
    for header in soup.find_all("div", class_="sectionheader"):
        if header.get_text(strip=True) == "Developer Reference":
            section = header.find_next_sibling("div", class_="section")
            if section:
                return section
    raise RuntimeError("Developer Reference not found")


def extract_category_name(details: Tag) -> str:
    summary = details.find("summary")
    div = summary.find("div") if summary else None

    if not div:
        return summary.get_text(strip=True) if summary else "unknown"

    span = div.find("span", class_="child-count")
    if span:
        span.extract()

    return div.get_text(strip=True)


def parse_class_methods(cls_block: Tag) -> list[dict]:
    methods: list[dict] = []

    for a in cls_block.select("ul > li > a[href]"):
        a_classes = set(a.get("class") or [])
        if is_skipped_link_classes(a_classes):
            continue

        href = (a.get("href") or "").strip()
        if not href.startswith("/gmod/"):
            continue

        methods.append(
            {
                "title": a.get_text(strip=True),
                "url": urljoin(BASE, href),
                "type": "method",
                "realms": extract_realm(a_classes),
            }
        )

    return maybe_sample(methods)


def parse_classlike_category(block: Tag) -> list[dict]:
    items: list[dict] = []

    for cls in block.find_all("details", class_="level2"):
        cls_classes = set(cls.get("class") or [])
        if is_skipped_link_classes(cls_classes):
            continue

        summary = cls.find("summary")
        a = summary.find("a", href=True) if summary else None
        if not a:
            continue

        href = (a.get("href") or "").strip()
        if not href.startswith("/gmod/"):
            continue

        t = extract_type(cls_classes)
        if t == "unknown":
            t = "class"

        items.append(
            {
                "title": a.get_text(strip=True),
                "url": urljoin(BASE, href),
                "type": t,
                "methods": parse_class_methods(cls),
            }
        )

    return maybe_sample(items)


def parse_enum_category(block: Tag) -> list[dict]:
    enums: list[dict] = []

    for a in block.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href.startswith("/gmod/"):
            continue

        a_classes = set(a.get("class") or [])
        if is_skipped_link_classes(a_classes):
            continue

        enums.append(
            {
                "title": a.get_text(strip=True),
                "url": urljoin(BASE, href),
                "type": "enum",
            }
        )

    return maybe_sample(enums)


def parse_structures_category(block: Tag) -> list[dict]:
    structs: list[dict] = []

    for a in block.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href.startswith("/gmod/"):
            continue

        a_classes = set(a.get("class") or [])
        if is_skipped_link_classes(a_classes):
            continue

        structs.append(
            {
                "title": a.get_text(strip=True),
                "url": urljoin(BASE, href),
                "type": "structure",
            }
        )

    return maybe_sample(structs)


def parse_regular_category(block: Tag, category: str) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()

    for a in block.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href.startswith("/gmod/"):
            continue

        a_classes = set(a.get("class") or [])

        parent = a.find_parent("details")
        if parent:
            a_classes |= set(parent.get("class") or [])

        if is_skipped_link_classes(a_classes):
            continue

        url = urljoin(BASE, href)
        if url in seen:
            continue
        seen.add(url)

        t = extract_type(a_classes)
        if category == "Game Events":
            t = "gameevent"

        items.append(
            {
                "title": a.get_text(strip=True),
                "url": url,
                "type": t,
                "realms": extract_realm(a_classes),
            }
        )

    return maybe_sample(items)


async def parse_dev_reference(session: aiohttp.ClientSession) -> dict[str, list[dict]]:
    soup = await fetch_soup(session, ROOT)
    section = find_dev_section(soup)

    result: dict[str, list[dict]] = {}

    for block in section.find_all("details", class_="level1", recursive=False):
        category = extract_category_name(block)

        if category in SKIP_CATEGORIES:
            continue

        if category in CLASSLIKE_CATEGORIES:
            result[category] = parse_classlike_category(block)
        elif category == "Enumerations":
            result[category] = parse_enum_category(block)
        elif category == "Structures":
            result[category] = parse_structures_category(block)
        else:
            result[category] = parse_regular_category(block, category)

    return result


def parse_description(soup: BeautifulSoup) -> list[str]:
    desc = soup.find("div", class_="function_description")
    if not desc:
        return []
    return [p.get_text(" ", strip=True) for p in desc.find_all("p")]


def parse_arguments(block: Tag) -> list[dict]:
    args_section = block.find("div", class_="function_arguments")
    if not args_section:
        return []

    args: list[dict] = []

    for arg in args_section.find_all("div", recursive=False):
        default_tag = arg.find("span", class_="default")

        default = None
        if default_tag:
            default = default_tag.get_text(" ", strip=True)
            default = default.lstrip("=").strip()

        args.append(
            {
                "type": text_clean(arg.find("a", class_="link-page")),
                "name": text_clean(arg.find("span", class_="name")),
                "default": default,
                "description": text_clean(arg.find("div", class_="numbertagindent")),
            }
        )

    return args


def parse_returns(block: Tag) -> list[dict]:
    ret_section = block.find("div", class_="function_returns")
    if not ret_section:
        return []

    returns: list[dict] = []

    for r in ret_section.find_all("div", recursive=False):
        returns.append(
            {
                "type": text_clean(r.find("a", class_="link-page")),
                "description": text_clean(r.find("div", class_="numbertagindent")),
            }
        )

    return returns


async def parse_function_page(session: aiohttp.ClientSession, item: dict) -> None:
    soup = await fetch_soup(session, item["url"])

    block = soup.find("div", class_="libraryfunc") or soup.find(
        "div", class_="function"
    )
    if not block:
        return

    item["description"] = parse_description(soup)
    item["arguments"] = parse_arguments(block)
    item["returns"] = parse_returns(block)


def parse_enum_values(soup: BeautifulSoup) -> list[dict]:
    best_table: Tag | None = None
    best_score = 0

    for table in soup.find_all("table"):
        score = 0
        for tr in table.find_all("tr"):
            if len(tr.find_all("td")) >= 2:
                score += 1
        if score > best_score:
            best_score = score
            best_table = table

    if not best_table or best_score == 0:
        return []

    values: list[dict] = []

    for tr in best_table.find_all("tr"):
        cols = tr.find_all("td")
        if len(cols) < 2:
            continue

        name = text_clean(cols[0])
        value = text_clean(cols[1])
        description = text_clean(cols[2]) if len(cols) >= 3 else None

        values.append(
            {
                "name": name,
                "value": value,
                "description": description,
            }
        )

    return values


async def parse_enum_page(session: aiohttp.ClientSession, item: dict) -> None:
    soup = await fetch_soup(session, item["url"])
    item["description"] = parse_description(soup)
    item["values"] = parse_enum_values(soup)


def parse_callback_args(desc_tag: Tag) -> list[dict]:
    cb = desc_tag.find("div", class_="callback_args")
    if not cb:
        return []

    out: list[dict] = []

    for row in cb.find_all("div", recursive=False):
        idx_raw = text_clean(row.find("span", class_="numbertag"))
        arg_type = text_clean(row.find("a", class_="link-page"))
        name = text_clean(row.find("strong"))

        desc = row.get_text(" ", strip=True)

        for part in (idx_raw, arg_type, name):
            if part:
                desc = desc.replace(part, "", 1).strip()
        desc = desc.lstrip("-").strip()

        idx: int | None = None
        if idx_raw and idx_raw.isdigit():
            idx = int(idx_raw)

        out.append(
            {
                "index": idx,
                "type": arg_type,
                "name": name,
                "description": desc or None,
            }
        )

    return out


def extract_default_from_description(desc_tag: Tag) -> str | None:
    for s in desc_tag.find_all("strong"):
        t = (s.get_text(" ", strip=True) or "").lower()
        if "default" not in t:
            continue

        container = s.parent if isinstance(s.parent, Tag) else None

        if container:
            code = container.find("code")
            if code:
                return text_clean(code)

            text = container.get_text(" ", strip=True) or ""

            text2 = re.sub(r"(?i)\bdefault\s*:\s*", "", text, count=1).strip()
            if text2 and text2.lower() != "default":
                return text2 or None

    return None


def extract_member_description(desc_tag: Tag | None) -> str | None:
    if not desc_tag:
        return None

    clone_soup = BeautifulSoup(str(desc_tag), "lxml")
    clone = clone_soup.find("div", class_="description") or clone_soup.find("div")
    if not clone:
        return None

    for cb in clone.select("div.callback_args"):
        cb.decompose()

    for el in clone.find_all(["p", "div"]):
        t = (el.get_text(" ", strip=True) or "").lower()
        if "default:" in t or t.startswith("default"):
            if el is clone:
                continue
            el.decompose()

    for s in clone.find_all("strong"):
        t = (s.get_text(" ", strip=True) or "").lower()
        if "default" in t:
            nxt = s.find_next("code")
            s.decompose()
            if nxt:
                nxt.decompose()

    return text_clean(clone)


def parse_structure_members(soup: BeautifulSoup) -> list[dict]:
    members: list[dict] = []

    for m in soup.select("div.member"):
        anchor = m.find("a", class_="struct_anchor_link")
        name = (
            text_clean(anchor.find("strong"))
            if anchor
            else text_clean(m.find("strong"))
        )

        anchor_classes = set(anchor.get("class") or []) if anchor else set()
        realms = extract_realm(anchor_classes)

        type_tag = m.find("a", class_="link-page")
        field_type = text_clean(type_tag)

        desc_tag = m.find("div", class_="description")
        default = extract_default_from_description(desc_tag) if desc_tag else None
        callback_args = parse_callback_args(desc_tag) if desc_tag else []
        description = extract_member_description(desc_tag)

        entry: dict = {
            "name": name,
            "type": field_type,
            "realms": realms,
            "default": default,
            "description": description,
        }

        if callback_args:
            entry["callback_args"] = callback_args

        members.append(entry)

    return members


async def parse_structure_page(session: aiohttp.ClientSession, item: dict) -> None:
    soup = await fetch_soup(session, item["url"])
    item["description"] = parse_description(soup)
    item["fields"] = parse_structure_members(soup)


_DISP_FUNC = {
    "enum": parse_enum_page,
    "structure": parse_structure_page,
    "gameevent": parse_structure_page,
}


async def main() -> None:
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(
        headers=HEADERS,
        timeout=timeout,
        connector=connector,
    ) as session:
        dev = await parse_dev_reference(session)

        sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def handle_one(item: dict) -> str:
            async with sem:
                func = _DISP_FUNC.get(item["type"], parse_function_page)
                await func(session, item)
                return item["url"]

        tasks: list[asyncio.Task[str]] = []

        for _category, items in dev.items():
            for item in items:
                if "methods" in item:
                    for m in item.get("methods", []):
                        tasks.append(asyncio.create_task(handle_one(m)))
                else:
                    tasks.append(asyncio.create_task(handle_one(item)))

        total = len(tasks)
        done = 0

        for fut in asyncio.as_completed(tasks):
            url = await fut
            done += 1
            print(f"parsed [{done}/{total}]: {url}", flush=True)

        OUT.write_text(
            json.dumps(dev, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        print("Saved ->", OUT, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
