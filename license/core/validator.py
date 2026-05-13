import re

# Uzbekistan plate format: 2-digit region + 1 letter + 3 digits + 2 letters
# Examples: 01A123BC, 10K456MN, 50Z999AB
UZ_PLATE_RE = re.compile(r'^[0-9]{2}[A-Z][0-9]{3}[A-Z]{2}$')

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


def validate_plate(text: str) -> tuple[bool, str]:
    """
    Returns (is_valid, normalized_plate_text).
    Applies normalization and attempts common OCR correction before validating.
    """
    normalized = normalize(text)

    if UZ_PLATE_RE.match(normalized) and normalized[:2] in VALID_REGIONS:
        return True, normalized

    # Attempt OCR correction on the letter positions (index 2 and 6,7)
    # Region digits at 0-1 should stay digits; letter at 2; digits at 3-5; letters at 6-7
    corrected = _try_correct(normalized)
    if corrected and UZ_PLATE_RE.match(corrected) and corrected[:2] in VALID_REGIONS:
        return True, corrected

    return False, normalized


def _try_correct(text: str) -> str | None:
    """Attempt to fix a plate that is close to valid UZ format."""
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
