"""Generate contract names for different exchanges."""
from __future__ import annotations


def contract_names(variety: str, start_year: int = 2022, end_year: int = 2026) -> list[str]:
    """Generate contract names for a variety, handling exchange-specific naming.

    DCE: DCE.jd2305 (2-digit year)
    SHFE: SHFE.rb2305 (2-digit year)
    CZCE: CZCE.MA305 (1-digit year for 2023+, 2-digit for older)
    INE: INE.sc2305 (2-digit year)
    CFFEX: CFFEX.IF2305 (2-digit year)
    """
    prefix = variety
    exchange = variety.split(".")[0] if "." in variety else ""

    names: list[str] = []
    for y in range(start_year - 1, end_year + 2):
        for m in range(1, 13):
            if exchange.upper() == "CZCE":
                # CZCE uses single-digit year code since ~2021
                yd = y % 10
                names.append(f"{prefix}{yd}{m:02d}")
            else:
                # Standard 2-digit year
                names.append(f"{prefix}{y % 100:02d}{m:02d}")
    return names


def contract_name_to_standard(name: str) -> str:
    """Convert any exchange contract name to standard format with dots replaced."""
    return name.replace(".", "_")


def standard_to_contract_name(standard: str, exchange: str = "") -> str:
    """Convert standard format back to contract name."""
    # CZCE_MAtest -> CZCE.MAtest
    parts = standard.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}.{parts[1]}"
    return standard
