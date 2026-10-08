"""Text cleaning shared by training and serving."""
import re

RENAMES = {"Installations": "Installs & Demo", "Consumables": "Filters & Consumables"}

# Product names are masked: routing depends on the problem, not the appliance.
_PRODUCTS = ["air fryer", "fryer", "mixer grinder", "mixer", "water purifier", "purifier",
             "induction cooktop", "cooktop", "room heater", "heater", "ceiling fan", "fan",
             "robot vacuum", "vacuum", "machine", "product"]
_PROD_RE = re.compile(r"\b(" + "|".join(sorted(_PRODUCTS, key=len, reverse=True)) + r")\b")
_ID_RE = re.compile(r"\b(sr|ko)\d+\b")


def fix_mojibake(s: str) -> str:
    """Legacy Zoho rows have UTF-8 read as cp1252 ('Ã©' for 'é'). Repair when possible."""
    try:
        return s.encode("cp1252").decode("utf8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def clean(s: str) -> str:
    s = fix_mojibake(str(s)).lower()
    s = _ID_RE.sub(" ", s)          # request / order numbers carry no routing signal
    s = _PROD_RE.sub(" prod ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalise_team(name: str) -> str:
    return RENAMES.get(name, name)
