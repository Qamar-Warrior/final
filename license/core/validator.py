import re
from enum import Enum


class PlateType(str, Enum):
    PUBLIC = "public"          # 01Z552TB  — 2 digits + 1 letter + 3 digits + 2 letters
    GOVERNMENT = "government"  # 01007WHA  — 2 digits + 3 digits + 3 letters
    UNKNOWN = "unknown"


# Public plate:     region(2) + letter(1) + digits(3) + letters(2)  e.g. 01Z552TB
_PUBLIC_RE = re.compile(r'^[0-9]{2}[A-Z][0-9]{3}[A-Z]{2}$')

# Government/company plate: region(2) + digits(3) + letters(3)  e.g. 01007WHA
_GOVT_RE = re.compile(r'^[0-9]{2}[0-9]{3}[A-Z]{3}$')

# Official Uzbekistan region codes
VALID_REGIONS = {
    '01',  # Tashkent city
    '10',  # Tashkent region
    '20',  # Sirdaryo
    '25',  # Jizzakh
    '30',  # Namangan
    '35',  # Andijan
    '40',  # Fergana
    '50',  # Samarkand
    '55',  # Kashkadarya
    '60',  # Surkhandarya
    '65',  # Bukhara
    '70',  # Navoi
    '75',  # Khorezm
    '80',  # Karakalpakstan
}

# Common OCR misreads: letter -> digit or digit -> letter
_OCR_FIXES = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8',
    'G': '6', 'Q': '0',
}


def normalize(text: str) -> str:
    """Strip whitespace, dashes, dots and uppercase."""
    cleaned = re.sub(r'[\s\-\.]', '', text).upper()
    return cleaned


def validate_plate(text: str) -> tuple[bool, str, PlateType]:
    """
    Returns (is_valid, normalized_plate_text, plate_type).
    Tries both public and government formats, with OCR correction.
    """
    normalized = normalize(text)

    if _PUBLIC_RE.match(normalized) and normalized[:2] in VALID_REGIONS:
        return True, normalized, PlateType.PUBLIC

    if _GOVT_RE.match(normalized) and normalized[:2] in VALID_REGIONS:
        return True, normalized, PlateType.GOVERNMENT

    corrected_public = _try_correct_public(normalized)
    if corrected_public and _PUBLIC_RE.match(corrected_public) and corrected_public[:2] in VALID_REGIONS:
        return True, corrected_public, PlateType.PUBLIC

    corrected_govt = _try_correct_government(normalized)
    if corrected_govt and _GOVT_RE.match(corrected_govt) and corrected_govt[:2] in VALID_REGIONS:
        return True, corrected_govt, PlateType.GOVERNMENT

    # If OCR inserted one extra character anywhere (e.g. doubled a digit),
    # try all 9 single-character deletions to recover the real 8-char plate.
    if len(normalized) == 9:
        for i in range(9):
            candidate = normalized[:i] + normalized[i + 1:]
            is_valid, plate_text, plate_type = validate_plate(candidate)
            if is_valid:
                return True, plate_text, plate_type

    return False, normalized, PlateType.UNKNOWN


def _try_correct_public(text: str) -> str | None:
    """Attempt to fix a plate close to public format: [DD][L][DDD][LL]."""
    if len(text) != 8:
        return None

    chars = list(text)

    # Positions 0,1,3,4,5 must be digits
    for i in (0, 1, 3, 4, 5):
        if not chars[i].isdigit():
            fix = _OCR_FIXES.get(chars[i])
            if fix:
                chars[i] = fix
            else:
                return None

    # Positions 2,6,7 must be letters
    for i in (2, 6, 7):
        if not chars[i].isalpha():
            return None

    return ''.join(chars)


def extract_plate_from_text(text: str) -> tuple[bool, str, PlateType]:
    """
    Find a valid plate within a longer OCR string.

    Useful when OCR picks up plate-frame decorations alongside the number.
    Tries a direct match first, then slides an 8-char window through the string.
    """
    is_valid, plate_text, plate_type = validate_plate(text)
    if is_valid:
        return is_valid, plate_text, plate_type

    normalized = normalize(text)
    if len(normalized) <= 9:
        return False, normalized, PlateType.UNKNOWN

    for start in range(len(normalized) - 7):
        candidate = normalized[start:start + 8]
        is_valid, plate_text, plate_type = validate_plate(candidate)
        if is_valid:
            return True, plate_text, plate_type

    return False, normalized, PlateType.UNKNOWN


def _try_correct_government(text: str) -> str | None:
    """Attempt to fix a plate close to government format: [DD][DDD][LLL]."""
    if len(text) != 8:
        return None

    chars = list(text)

    # Positions 0-4 must be digits
    for i in range(5):
        if not chars[i].isdigit():
            fix = _OCR_FIXES.get(chars[i])
            if fix:
                chars[i] = fix
            else:
                return None

    # Positions 5,6,7 must be letters
    for i in (5, 6, 7):
        if not chars[i].isalpha():
            return None

    return ''.join(chars)
