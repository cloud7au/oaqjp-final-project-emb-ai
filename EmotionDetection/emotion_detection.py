
"""Emotion detection using IBM Watson NLP."""

import requests


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion."""

    empty_result = {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None
    }

    if not isinstance(text_to_analyze, str):
        return empty_result.copy()

    if not text_to_analyze.strip():
        return empty_result.copy()

    url = (
        "https://sn-watson-emotion.labs.skills.network"
        "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    headers = {
        "grpc-metadata-mm-model-id":
            "emotion_aggregated-workflow_lang_en_stock"
    }

    input_json = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    try:
        response = requests.post(
            url,
            json=input_json,
            headers=headers,
            timeout=30
        )

        if response.status_code == 400:
            return empty_result.copy()

        response.raise_for_status()
        formatted_response = response.json()

        emotions = formatted_response[
            "emotionPredictions"
        ][0]["emotion"]

        emotion_scores = {
            "anger": emotions["anger"],
            "disgust": emotions["disgust"],
            "fear": emotions["fear"],
            "joy": emotions["joy"],
            "sadness": emotions["sadness"]
        }

        dominant_emotion = max(
            emotion_scores,
            key=emotion_scores.get
        )

        emotion_scores["dominant_emotion"] = dominant_emotion

        return emotion_scores

    except (
        requests.exceptions.RequestException,
        ValueError,
        KeyError,
        IndexError,
        TypeError
    ):
        return empty_result.copy()
