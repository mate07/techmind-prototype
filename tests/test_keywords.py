import numpy as np
import pytest
from scipy.sparse import csr_matrix

from services.keywords import extract_keywords


pytestmark = pytest.mark.unit


class KeywordVectorizer:
    def transform(self, values):
        return csr_matrix([[0.2, 0.9, 0.5]])

    def get_feature_names_out(self):
        return np.array(["api", "flask", "python"])


def test_empty_text_has_no_keywords():
    assert extract_keywords("   ") == []


def test_frequency_fallback_filters_stopwords_and_respects_limit():
    result = extract_keywords("Python python con API api api C++ programación", top_n=3)
    assert result == ["api", "python", "c++"]


def test_vectorizer_orders_by_weight_and_respects_limit():
    assert extract_keywords("contenido", KeywordVectorizer(), top_n=2) == ["flask", "python"]
