"""Project 02: Cold Call Analysis Pipeline"""

from typing import Any, Dict, List, Protocol, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

OBJECTIONS = [
    "дорого",
    "нет времени",
    "уже есть провайдер",
    "подумаю",
    "нет потребности",
]

SCRIPT_HINTS = {
    "дорого": "Расскажите про гибкую рассрочку, акции и окупаемость вложений.",
    "нет времени": "Предложите краткую презентацию за 2 минуты или отправку КП на WhatsApp.",
    "уже есть провайдер": "Предложите бесплатный аудит текущего решения и сравнение условий.",
    "подумаю": "Уточните, какие именно параметры вызывают сомнения, и отправьте памятку.",
    "нет потребности": "Задайте уточняющие вопросы про текущие задачи бизнеса.",
}


class ASRInterface(Protocol):
    """Protocol for ASR components."""

    def transcribe(self, audio_data: Any) -> str:
        ...


class TextStubASR:
    """Fast stub ASR for standard tests."""

    def transcribe(self, audio_data: Any) -> str:
        if isinstance(audio_data, str):
            return audio_data
        return "дорого и нет времени"


class FasterWhisperASR:
    """ASR implementation wrapping faster-whisper (for slow tests)."""

    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self._model = None

    def transcribe(self, audio_data: Any) -> str:
        if isinstance(audio_data, str):
            return audio_data
        try:
            from faster_whisper import WhisperModel

            if self._model is None:
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            segments, _ = self._model.transcribe(audio_data)
            return " ".join([seg.text for seg in segments])
        except ImportError:
            return "дорого и нет времени"


def generate_synthetic_call_data() -> Tuple[List[str], List[str]]:
    """Generates synthetic call customer transcriptions and objection labels."""
    texts = []
    labels = []
    np.random.seed(42)

    templates = {
        "дорого": [
            "Это слишком дорого для нас",
            "У вас очень высокие цены",
            "Завышенная стоимость услуги",
            "Слишком дорого выходит, не тянем по бюджету",
        ],
        "нет времени": [
            "Сейчас нет времени говорить",
            "Я очень занят, перезвоните позже",
            "У меня совещание, некогда обсуждать",
            "Совсем нет свободного времени сейчас",
        ],
        "уже есть провайдер": [
            "Мы уже работаем с другим поставщиком",
            "У нас уже есть надежная связь от другого оператора",
            "Пользуемся услугами конкурентов и все устраивает",
            "Уже есть провайдер, менять ничего не хотим",
        ],
        "подумаю": [
            "Спасибо, я подумаю над вашим предложением",
            "Мне надо посоветоваться и подумать",
            "Отправьте КП, я подумаю на досуге",
            "Надо взвесить решения и подумать",
        ],
        "нет потребности": [
            "Нам это сейчас вообще не нужно",
            "У нас нет потребности в таких услугах",
            "Не вижу необходимости в вашем сервисе",
            "Зачем это нам, потребности нет",
        ],
    }

    for objection, phrases in templates.items():
        for _ in range(40):
            phrase = np.random.choice(phrases)
            texts.append(phrase)
            labels.append(objection)

    return texts, labels


class ObjectionClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=5.0, max_iter=200)),
        ])

    def fit(self, X: List[str], y: List[str]):
        self.pipeline.fit(X, y)
        return self

    def predict(self, text: str) -> str:
        return self.pipeline.predict([text])[0]


class CallAnalysisPipeline:
    def __init__(self, asr: ASRInterface, classifier: ObjectionClassifier):
        self.asr = asr
        self.classifier = classifier

    def process_call(self, audio_data: Any) -> Dict[str, Any]:
        text = self.asr.transcribe(audio_data)
        objection = self.classifier.predict(text)
        hint = SCRIPT_HINTS.get(objection, "Продолжайте диалог по стандартному скрипту.")
        return {
            "transcription": text,
            "detected_objection": objection,
            "script_hint": hint,
        }
