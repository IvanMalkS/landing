import importlib

parser_mod = importlib.import_module("projects.09_query_ner.parser")
QueryFilterParser = parser_mod.QueryFilterParser
generate_synthetic_queries = parser_mod.generate_synthetic_queries


def test_query_filter_exact_match_accuracy_above_90():
    dataset = generate_synthetic_queries(n_samples=120, seed=42)
    parser = QueryFilterParser()

    exact_matches = 0
    for query, expected_filters in dataset:
        parsed_filters = parser.parse_query(query)
        if parsed_filters == expected_filters:
            exact_matches += 1

    accuracy = exact_matches / len(dataset)
    assert accuracy >= 0.90, f"Exact match filter accuracy {accuracy:.3f} is lower than 0.90"


def test_unknown_words_and_unseen_noise_do_not_break_parser():
    parser = QueryFilterParser()

    query = "у метро с парковкой до 3500 несуществующееслово фывапролг123"
    parsed = parser.parse_query(query)

    assert parsed.get("near_metro") is True
    assert parsed.get("parking") is True
    assert parsed.get("price_max") == 3500


def test_specific_example_from_spec():
    parser = QueryFilterParser()
    query = "у метро, с парковкой, до 3000"
    parsed = parser.parse_query(query)

    assert parsed == {"near_metro": True, "parking": True, "price_max": 3000}
