from __future__ import annotations

import re
import unicodedata


STOPWORDS = {
    "ai", "anh", "ay", "ban", "bang", "bao", "bi", "cac", "cai", "can", "chi",
    "cho", "co", "con", "cua", "da", "dang", "day", "de", "den", "di", "do",
    "duoc", "duoi", "gi", "giua", "hay", "hon", "khi", "khong", "la", "lai",
    "lam", "may", "mot", "nao", "nay", "neu", "ngay", "nhieu", "nhu", "nhung",
    "o", "phai", "qua", "ra", "rang", "sau", "se", "so", "tai", "the", "thi",
    "theo", "tren", "trong", "truoc", "tu", "va", "vao", "ve", "vi", "voi",
}

DATE_TOKENS = {"ngay", "thang", "nam"}


def normalize_text(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    value = unicodedata.normalize("NFD", value.lower())
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[\w%]+", normalize_text(value), flags=re.UNICODE)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def raw_tokens(value: str) -> list[str]:
    return re.findall(r"[\w%]+", normalize_text(value), flags=re.UNICODE)


__all__ = ["DATE_TOKENS", "STOPWORDS", "normalize_text", "raw_tokens", "tokenize"]
