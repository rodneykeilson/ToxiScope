"""
ToxiScope Text Preprocessing for English Toxicity Classification

This module provides text normalization and cleaning specifically designed
for English gaming community comments from Reddit.

Features:
- URL and mention handling
- Emoji processing
- Gaming abbreviation expansion
- Profanity pattern normalization
- Whitespace and character normalization
"""

import re
import string
from typing import List, Optional, Dict
import unicodedata


# Common gaming/internet abbreviations and their expansions
ABBREVIATIONS: Dict[str, str] = {
    # Gaming terms
    "gg": "good game",
    "wp": "well played",
    "ez": "easy",
    "noob": "newbie",
    "n00b": "newbie",
    "afk": "away from keyboard",
    "brb": "be right back",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "tbh": "to be honest",
    "ngl": "not gonna lie",
    "smh": "shaking my head",
    "ffs": "for fuck's sake",
    "stfu": "shut the fuck up",
    "gtfo": "get the fuck out",
    "lmao": "laughing my ass off",
    "lmfao": "laughing my fucking ass off",
    "rofl": "rolling on floor laughing",
    "omg": "oh my god",
    "wtf": "what the fuck",
    "wth": "what the hell",
    "idk": "i don't know",
    "idc": "i don't care",
    "idgaf": "i don't give a fuck",
    "af": "as fuck",
    "rn": "right now",
    "nvm": "nevermind",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "ty": "thank you",
    "np": "no problem",
    "gl": "good luck",
    "hf": "have fun",
    "glhf": "good luck have fun",
    "ggwp": "good game well played",
    "pog": "play of the game",
    "poggers": "exciting",
    "kappa": "sarcasm",
    "copium": "coping mechanism",
    "copege": "desperate coping",
    "sadge": "sad",
    "pepe": "frog meme",
    "pepega": "stupid",
    "monkas": "nervous",
    "rekt": "wrecked",
    "pwned": "owned",
    "oof": "expression of discomfort",
    "kek": "laugh",
    "lul": "laugh",
    "xd": "laughing face",
    "xdd": "laughing face",
    
    # Game-specific
    "cs": "counter strike",
    "cs2": "counter strike 2",
    "dota": "defense of the ancients",
    "lol": "league of legends",
    "ow": "overwatch",
    "cod": "call of duty",
    "fn": "fortnite",
    "apex": "apex legends",
    "val": "valorant",
    "mlbb": "mobile legends bang bang",
    "ml": "mobile legends",
    "pubg": "playerunknown's battlegrounds",
    
    # Role-specific
    "adc": "attack damage carry",
    "supp": "support",
    "jg": "jungle",
    "jungler": "jungle player",
    "mid": "middle lane",
    "top": "top lane",
    "bot": "bottom lane",
    "carry": "main damage dealer",
    "tank": "defensive player",
    "healer": "support healer",
    "dps": "damage per second",
}

# Obfuscated profanity patterns (leetspeak, symbols, etc.)
OBFUSCATION_PATTERNS: List[tuple] = [
    # Letter substitutions
    (r'[@4]', 'a'),
    (r'[3€]', 'e'),
    (r'[1!|]', 'i'),
    (r'[0ø]', 'o'),
    (r'[$5]', 's'),
    (r'[7+]', 't'),
    # Common obfuscations
    (r'f+[*_\-\.]+c+k+', 'fuck'),
    (r'sh+[*_\-\.]+t+', 'shit'),
    (r'b+[*_\-\.]+t+c+h+', 'bitch'),
    (r'a+[*_\-\.]+s+', 'ass'),
    (r'n+[*_\-\.]+g+', 'nig'),  # Partial for detection
    (r'd+[*_\-\.]+c+k+', 'dick'),
    (r'c+[*_\-\.]+n+t+', 'cunt'),
    # Repeated letters (e.g., "stuuuupid")
    (r'(.)\1{2,}', r'\1\1'),
]


class EnglishTextNormalizer:
    """
    Text normalizer for English gaming community comments.
    
    Designed for toxicity classification - preserves semantic content
    while normalizing variations in spelling and formatting.
    """
    
    def __init__(
        self,
        expand_abbreviations: bool = True,
        normalize_obfuscation: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        lowercase: bool = True,
        remove_emojis: bool = False,  # Keep for context
        min_length: int = 2,
    ):
        """
        Initialize normalizer.
        
        Args:
            expand_abbreviations: Expand gaming/internet abbreviations
            normalize_obfuscation: Normalize leetspeak and symbol obfuscation
            remove_urls: Remove URLs from text
            remove_mentions: Remove @mentions and u/mentions
            lowercase: Convert to lowercase
            remove_emojis: Remove emoji characters
            min_length: Minimum text length after normalization
        """
        self.expand_abbreviations = expand_abbreviations
        self.normalize_obfuscation = normalize_obfuscation
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.lowercase = lowercase
        self.remove_emojis = remove_emojis
        self.min_length = min_length
        
        # Compile regex patterns
        self._url_pattern = re.compile(
            r'https?://\S+|www\.\S+|[\w\-]+\.(com|org|net|io|gg|tv)\S*',
            re.IGNORECASE
        )
        self._mention_pattern = re.compile(r'[@/]?u/[\w\-]+|@[\w\-]+', re.IGNORECASE)
        self._subreddit_pattern = re.compile(r'r/[\w\-]+', re.IGNORECASE)
        self._whitespace_pattern = re.compile(r'\s+')
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
    
    def normalize(self, text: str) -> str:
        """
        Normalize text for toxicity classification.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        if not text:
            return ""
        
        # Remove URLs
        if self.remove_urls:
            text = self._url_pattern.sub(' ', text)
        
        # Remove mentions but keep subreddit references
        if self.remove_mentions:
            text = self._mention_pattern.sub(' ', text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove emojis if requested
        if self.remove_emojis:
            text = self._emoji_pattern.sub(' ', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Normalize obfuscation (leetspeak, symbols)
        if self.normalize_obfuscation:
            text = self._normalize_obfuscation(text)
        
        # Expand abbreviations
        if self.expand_abbreviations:
            text = self._expand_abbreviations(text)
        
        # Normalize whitespace
        text = self._whitespace_pattern.sub(' ', text).strip()
        
        # Check minimum length
        if len(text) < self.min_length:
            return ""
        
        return text
    
    def _normalize_obfuscation(self, text: str) -> str:
        """Normalize leetspeak and symbol obfuscation."""
        for pattern, replacement in OBFUSCATION_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common gaming/internet abbreviations."""
        words = text.split()
        expanded = []
        for word in words:
            # Clean punctuation for matching
            clean_word = word.strip(string.punctuation).lower()
            if clean_word in ABBREVIATIONS:
                expanded.append(ABBREVIATIONS[clean_word])
            else:
                expanded.append(word)
        return ' '.join(expanded)
    
    def __call__(self, text: str) -> str:
        """Allow using normalizer as callable."""
        return self.normalize(text)


def normalize_text(text: str) -> str:
    """
    Simple normalization function for compatibility.
    
    Args:
        text: Raw text
        
    Returns:
        Normalized text
    """
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = ' '.join(text.split())
    
    return text


def batch_normalize(
    texts: List[str],
    normalizer: Optional[EnglishTextNormalizer] = None,
) -> List[str]:
    """
    Normalize a batch of texts.
    
    Args:
        texts: List of raw texts
        normalizer: Optional pre-configured normalizer
        
    Returns:
        List of normalized texts
    """
    if normalizer is None:
        normalizer = EnglishTextNormalizer()
    
    return [normalizer(text) for text in texts]


def create_normalizer_for_training() -> EnglishTextNormalizer:
    """
    Create normalizer optimized for training.
    
    More aggressive normalization to reduce vocabulary.
    """
    return EnglishTextNormalizer(
        expand_abbreviations=True,
        normalize_obfuscation=True,
        remove_urls=True,
        remove_mentions=True,
        lowercase=True,
        remove_emojis=False,  # Keep for sentiment signal
        min_length=2,
    )


def create_normalizer_for_inference() -> EnglishTextNormalizer:
    """
    Create normalizer optimized for inference.
    
    Less aggressive to preserve user intent.
    """
    return EnglishTextNormalizer(
        expand_abbreviations=False,  # Keep original form
        normalize_obfuscation=True,  # Still catch obfuscated slurs
        remove_urls=True,
        remove_mentions=True,
        lowercase=True,
        remove_emojis=False,
        min_length=1,
    )


if __name__ == "__main__":
    # Test the normalizer
    normalizer = EnglishTextNormalizer()
    
    test_cases = [
        "You're such a n00b lmao git gud",
        "stfu and gtfo you f***ing idiot @username",
        "Check out https://youtube.com/watch?v=xyz this is POGGERS",
        "ggwp ez game ez life 😂😂😂",
        "r/DotA2 is the best subreddit tbh ngl",
        "This is $h1t and you know it",
        "STOPPPPP BEING SO STUUUUPID",
        "idc what you think, you're trash at cs2",
    ]
    
    print("English Text Normalizer Test")
    print("=" * 60)
    
    for text in test_cases:
        normalized = normalizer(text)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print("-" * 60)
