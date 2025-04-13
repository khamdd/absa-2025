import re
# import string
import nltk
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
# nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class EnglishTextCleaner:
    # def __init__(self):
    #     self.stop_words = set(stopwords.words('english'))

    def remove_html_tags(self, text):
        return re.sub(r'<[^>]+>', '', text)

    def remove_urls(self, text):
        return re.sub(r'http\S+|www\.\S+', '', text)

    def remove_emails(self, text):
        return re.sub(r'\S+@\S+\.\S+', '', text)

    def remove_special_characters(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    def remove_extra_whitespace(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    # def remove_stopwords(self, text):
    #     tokens = word_tokenize(text)
    #     return ' '.join([word for word in tokens if word.lower() not in self.stop_words])

    def clean_text(self, text):
        text = text.lower()
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_special_characters(text)
        text = self.remove_extra_whitespace(text)
        # text = self.remove_stopwords(text)
        return text

class EnglishTextNormalizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_text(self, text):
        tokens = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in tokens])

    def normalize_text(self, text):
        text = self.lemmatize_text(text)
        return text

class EnglishTextPreprocessor:
    def __init__(self):
        self.cleaner = EnglishTextCleaner()
        self.normalizer = EnglishTextNormalizer()

    def preprocess(self, text):
        text = self.cleaner.clean_text(text)
        text = self.normalizer.normalize_text(text)
        return text
    
    def process_batch(self, texts):
        texts = [self.preprocess(text) for text in texts]
        return [self.preprocess(text) for text in texts]

if __name__ == "__main__":
    # Example usage
    processor = EnglishTextPreprocessor()
    sample_text = "HELLO, my name is JOHN DOE!!! You can reach me at John.Doe@example.com, or visit my website at https://www.johndoe.com for more information. BTW, did you know that the price of the item was $99.99 (!!!) on 2025-04-13, but NOW it’s just $49.99?!? Amazing, right? ALSO, check out THIS URL: <a href='https://example.com'>CLICK HERE</a>."
    processed_text = processor.preprocess(sample_text)
    print("Original text:", sample_text)
    print("Processed text:", processed_text)
