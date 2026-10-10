"""Project 09: Search Query Parsing into Structural Filters (NER & Rule Parser)"""

import re
from typing import Any, Dict, List, Tuple

import spacy


class QueryFilterParser:
    """Parses natural language rental queries into structured json filter objects."""

    def __init__(self):
        self.nlp = spacy.blank("ru")

    def parse_query(self, query: str) -> Dict[str, Any]:
        doc = self.nlp(query.lower())
        text = doc.text

        filters: Dict[str, Any] = {}

        # 1. Location / Near Metro
        if "метро" in text or "возле метро" in text or "у метро" in text or "рядом с метро" in text:
            filters["near_metro"] = True

        # 2. Amenities / Features
        if "парковкой" in text or "парковка" in text or "гараж" in text or "с машиноместом" in text:
            filters["parking"] = True

        if "балконом" in text or "балкон" in text or "лоджия" in text:
            filters["has_balcony"] = True

        if "животными" in text or "питомцами" in text or "можно с кошкой" in text or "собакой" in text:
            filters["pets_allowed"] = True

        # 3. Price max extraction (e.g., "до 3000", "до 5000 руб", "не дороже 4000")
        price_match = re.search(r"(?:до|дешевле|не дороже)\s*(\d{3,6})", text)
        if price_match:
            filters["price_max"] = int(price_match.group(1))

        # Price min extraction (e.g., "от 1500", "дороже 2000")
        price_min_match = re.search(r"(?:от|дороже)\s*(\d{3,6})", text)
        if price_min_match:
            filters["price_min"] = int(price_min_match.group(1))

        # 4. Rooms extraction (e.g., "1-комнатная", "2 комнаты", "студия")
        if "студи" in text:
            filters["rooms"] = 0
        else:
            room_match = re.search(r"(\d)\s*(?:комнат|комн|ккв)", text)
            if room_match:
                filters["rooms"] = int(room_match.group(1))

        return filters


def generate_synthetic_queries(n_samples: int = 120, seed: int = 42) -> List[Tuple[str, Dict[str, Any]]]:
    """Generates synthetic search queries and exact expected ground-truth filter dictionaries."""
    import numpy as np

    np.random.seed(seed)

    metro_phrases = ["у метро", "рядом с метро", "возле метро", ""]
    parking_phrases = ["с парковкой", "с бесплатной парковкой", ""]
    price_phrases = ["до 3000", "до 5000", "до 10000", ""]
    pets_phrases = ["можно с животными", "с питомцами", ""]
    unknown_noise = ["абракадабра", "быстро", "срочно", "недорого", "люксовый вариант"]

    dataset = []
    for i in range(n_samples):
        m = np.random.choice(metro_phrases)
        p = np.random.choice(parking_phrases)
        pr = np.random.choice(price_phrases)
        pt = np.random.choice(pets_phrases)
        noise = np.random.choice(unknown_noise) if np.random.rand() < 0.3 else ""

        tokens = [t for t in [m, p, pr, pt, noise] if t]
        np.random.shuffle(tokens)
        query = " ".join(tokens) if tokens else "квартира посуточно"

        expected = {}
        if m:
            expected["near_metro"] = True
        if p:
            expected["parking"] = True
        if pt:
            expected["pets_allowed"] = True
        if pr:
            expected["price_max"] = int(pr.split()[-1])

        dataset.append((query, expected))

    return dataset
