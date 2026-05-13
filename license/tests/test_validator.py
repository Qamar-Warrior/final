import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.validator import validate_plate, normalize

# ------------------------------------------------------------------
# normalize()
# ------------------------------------------------------------------

def test_normalize_strips_spaces():
    assert normalize(" 01 A 123 BC ") == "01A123BC"

def test_normalize_strips_dashes():
    assert normalize("01-A-123-BC") == "01A123BC"

def test_normalize_uppercases():
    assert normalize("01a123bc") == "01A123BC"

def test_normalize_strips_dots():
    assert normalize("01.A.123.BC") == "01A123BC"

# ------------------------------------------------------------------
# validate_plate() — valid inputs
# ------------------------------------------------------------------

def test_valid_standard():
    ok, text = validate_plate("01A123BC")
    assert ok is True
    assert text == "01A123BC"

def test_valid_max_region():
    ok, text = validate_plate("14Z999AB")
    assert ok is True
    assert text == "14Z999AB"

def test_valid_with_spaces():
    ok, text = validate_plate(" 01 A 123 BC ")
    assert ok is True
    assert text == "01A123BC"

def test_valid_lowercase_normalized():
    ok, text = validate_plate("01a123bc")
    assert ok is True
    assert text == "01A123BC"

# ------------------------------------------------------------------
# validate_plate() — invalid inputs
# ------------------------------------------------------------------

def test_invalid_too_short():
    ok, _ = validate_plate("01A12BC")
    assert ok is False

def test_invalid_too_long():
    ok, _ = validate_plate("01A1234BC")
    assert ok is False

def test_invalid_wrong_region_format():
    ok, _ = validate_plate("1A123BC")    # only 1 region digit
    assert ok is False

def test_invalid_empty():
    ok, _ = validate_plate("")
    assert ok is False

def test_invalid_all_digits():
    ok, _ = validate_plate("01112345")
    assert ok is False

def test_invalid_letters_in_digit_positions():
    ok, _ = validate_plate("AAA123BC")
    assert ok is False

# ------------------------------------------------------------------
# validate_plate() — OCR correction
# ------------------------------------------------------------------

def test_ocr_correction_O_to_0():
    # 'O' at digit position should be corrected to '0'
    ok, text = validate_plate("O1A123BC")
    assert ok is True
    assert text == "01A123BC"

def test_ocr_correction_I_to_1():
    ok, text = validate_plate("0IA123BC")
    assert ok is True
    assert text == "01A123BC"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
