import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.validator import PlateType, validate_plate, normalize

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
    ok, text, _ = validate_plate("01A123BC")
    assert ok is True
    assert text == "01A123BC"

def test_valid_karakalpakstan_region():
    ok, text, _ = validate_plate("80Z999AB")
    assert ok is True
    assert text == "80Z999AB"

def test_valid_with_spaces():
    ok, text, _ = validate_plate(" 01 A 123 BC ")
    assert ok is True
    assert text == "01A123BC"

def test_valid_lowercase_normalized():
    ok, text, _ = validate_plate("01a123bc")
    assert ok is True
    assert text == "01A123BC"

# ------------------------------------------------------------------
# validate_plate() — invalid inputs
# ------------------------------------------------------------------

def test_invalid_too_short():
    ok, _text, _type = validate_plate("01A12BC")
    assert ok is False

def test_invalid_too_long():
    ok, _text, _type = validate_plate("01A1234BC")
    assert ok is False

def test_invalid_wrong_region_format():
    ok, _text, _type = validate_plate("1A123BC")    # only 1 region digit
    assert ok is False

def test_invalid_empty():
    ok, _text, _type = validate_plate("")
    assert ok is False

def test_invalid_all_digits():
    ok, _text, _type = validate_plate("01112345")
    assert ok is False

def test_invalid_letters_in_digit_positions():
    ok, _text, _type = validate_plate("AAA123BC")
    assert ok is False

# ------------------------------------------------------------------
# validate_plate() — OCR correction
# ------------------------------------------------------------------

def test_ocr_correction_O_to_0():
    # 'O' at digit position should be corrected to '0'
    ok, text, _ = validate_plate("O1A123BC")
    assert ok is True
    assert text == "01A123BC"

def test_ocr_correction_I_to_1():
    ok, text, _ = validate_plate("0IA123BC")
    assert ok is True
    assert text == "01A123BC"


# ------------------------------------------------------------------
# PlateType classification
# ------------------------------------------------------------------

def test_public_plate_type():
    ok, text, plate_type = validate_plate("01Z552TB")
    assert ok is True
    assert text == "01Z552TB"
    assert plate_type == PlateType.PUBLIC

def test_government_plate_type():
    ok, text, plate_type = validate_plate("01007WHA")
    assert ok is True
    assert text == "01007WHA"
    assert plate_type == PlateType.GOVERNMENT

def test_government_plate_various():
    ok, text, plate_type = validate_plate("10123ABC")
    assert ok is True
    assert plate_type == PlateType.GOVERNMENT

def test_invalid_plate_type_unknown():
    ok, _text, plate_type = validate_plate("XXXXXXXX")
    assert ok is False
    assert plate_type == PlateType.UNKNOWN

def test_government_ocr_correction():
    # 'O' at digit position corrected to '0'
    ok, text, plate_type = validate_plate("O1007WHA")
    assert ok is True
    assert text == "01007WHA"
    assert plate_type == PlateType.GOVERNMENT


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
