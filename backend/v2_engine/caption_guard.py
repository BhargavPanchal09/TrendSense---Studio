"""Bound generated image captions before returning them to the dashboard."""
import re


def short_visual_caption(text, max_words=24):
    cleaned = " ".join((text or "").replace("\x00", " ").split())
    if not cleaned:
        return "No image description was returned."
    placeholder = cleaned.strip(' .\"').casefold()
    if placeholder in {'image uploaded', 'unable to identify visible content'}:
        return "The vision model did not provide a usable description."
    first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
    words = first_sentence.split()
    if len(words) > max_words:
        first_sentence = " ".join(words[:max_words]).rstrip(",;:") + "..."
    return first_sentence
