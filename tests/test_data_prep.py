"""Unit tests for src.data_prep — runs without network."""

from src.data_prep import Sample, is_mostly_devanagari, is_valid, dedupe, shingles


def test_is_mostly_devanagari_true_marathi():
    # "This is a Marathi sentence." in Marathi.
    assert is_mostly_devanagari("हे एक मराठी वाक्य आहे.")


def test_is_mostly_devanagari_true_hindi():
    # Hindi is also Devanagari — script-only check cannot distinguish them.
    # We accept Devanagari here and trust source dataset metadata to filter for mr.
    assert is_mostly_devanagari("यह एक हिंदी वाक्य है।")


def test_is_mostly_devanagari_false_english():
    assert not is_mostly_devanagari("This is an English sentence.")


def test_is_mostly_devanagari_false_gujarati():
    # Gujarati is a different Indic script and must not pass the Devanagari filter.
    assert not is_mostly_devanagari("આ એક ગુજરાતી વાક્ય છે.")


def test_is_mostly_devanagari_empty():
    assert not is_mostly_devanagari("")


def test_is_valid_rejects_short():
    s = Sample(instruction="हाय", response="आहे", source="x")
    assert not is_valid(s)


def test_is_valid_accepts_normal():
    s = Sample(
        instruction="महाराष्ट्राची राजधानी कोणती आहे?",
        response="महाराष्ट्राची राजधानी मुंबई आहे.",
        source="test",
    )
    assert is_valid(s)


def test_dedupe_removes_near_duplicates():
    s1 = Sample("महाराष्ट्राची राजधानी कोणती आहे?", "मुंबई आहे.", "a")
    s2 = Sample("महाराष्ट्राची राजधानी कोणती आहे?", "मुंबई आहे.", "b")  # exact dup
    s3 = Sample("भारताची राजधानी कोणती आहे?", "नवी दिल्ली आहे.", "c")  # different
    kept = dedupe([s1, s2, s3])
    assert len(kept) == 2


def test_shingles_short_input():
    assert shingles("hi", k=5) == {"hi"}


def test_chatml_format():
    s = Sample("नमस्कार, कसे आहात?", "मी ठीक आहे.", "test")
    text = s.chatml_text()
    assert "<|im_start|>user" in text
    assert "<|im_start|>assistant" in text
    assert "<|im_end|>" in text
    assert "नमस्कार, कसे आहात?" in text
