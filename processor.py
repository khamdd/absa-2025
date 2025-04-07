import re
from collections import Counter
from spellchecker import SpellChecker

class SimpleTextProcessor:
    def __init__(self):
        self.spell_checker = SpellChecker()

    def remove_punctuation(self, text):
        """Remove punctuation and digits."""
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def to_lowercase(self, text):
        """Convert text to lowercase."""
        return text.lower()

    def remove_extra_spaces(self, text):
        """Remove multiple spaces."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def correct_spelling(self, text):
        """Correct common spelling mistakes."""
        words = text.split()
        corrected_words = [self.spell_checker.correction(word) for word in words]
        return ' '.join(corrected_words)

    def preprocess_text(self, text):
        """Process text: Clean, normalize and correct spelling."""
        text = self.remove_punctuation(text)
        text = self.to_lowercase(text)
        text = self.remove_extra_spaces(text)
        text = self.correct_spelling(text)
        return text
