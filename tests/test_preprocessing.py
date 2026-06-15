from preprocessing import preprocess_text


def test_lowercases_and_removes_stopwords():
    assert preprocess_text("The QUICK brown fox") == "quick brown fox"


def test_masks_urls():
    assert "url" in preprocess_text("visit http://evil.com now")


def test_masks_emails():
    assert "email" in preprocess_text("contact me at bob@example.com")


def test_removes_digits_and_punctuation():
    assert preprocess_text("Order #12345!!!") == "order"


def test_handles_non_string_input():
    assert preprocess_text(None) == ""
