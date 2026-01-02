"""
ToxiScope: Data Augmentation for English Text

Implements various augmentation strategies for handling class imbalance
and improving model performance in toxicity classification.

References:
- Wei & Zou (2019) "EDA: Easy Data Augmentation Techniques"
- Zhang et al. (2017) "MixUp: Beyond Empirical Risk Minimization"
"""

import random
import re
from typing import List, Optional, Tuple, Dict
from collections import Counter


class TextAugmenter:
    """
    Text augmentation for English gaming community text.
    
    Augmentation strategies:
    1. Synonym Replacement - Replace words with synonyms
    2. Random Deletion - Randomly delete words
    3. Random Swap - Swap adjacent words
    4. Random Insertion - Insert random synonyms
    5. Character Noise - Add typos
    """
    
    # Common word synonyms for gaming/internet text
    SYNONYMS: Dict[str, List[str]] = {
        # Positive words
        "good": ["great", "nice", "awesome", "excellent", "solid"],
        "great": ["good", "amazing", "fantastic", "excellent"],
        "awesome": ["amazing", "incredible", "fantastic", "sick"],
        "nice": ["good", "cool", "sweet", "solid"],
        "cool": ["nice", "awesome", "sick", "dope"],
        "amazing": ["incredible", "awesome", "fantastic", "insane"],
        "best": ["greatest", "top", "finest", "goat"],
        "love": ["adore", "enjoy", "like"],
        "like": ["enjoy", "love", "appreciate"],
        
        # Negative words
        "bad": ["terrible", "awful", "trash", "garbage"],
        "terrible": ["awful", "horrible", "bad", "trash"],
        "awful": ["terrible", "horrible", "bad"],
        "trash": ["garbage", "terrible", "bad", "useless"],
        "garbage": ["trash", "junk", "waste"],
        "stupid": ["dumb", "idiotic", "moronic", "brainless"],
        "dumb": ["stupid", "idiotic", "dense"],
        "hate": ["despise", "loathe", "detest"],
        "annoying": ["irritating", "frustrating", "obnoxious"],
        "boring": ["dull", "tedious", "lame"],
        
        # Gaming terms
        "player": ["gamer", "user"],
        "game": ["match", "round"],
        "team": ["squad", "crew", "group"],
        "win": ["victory", "success"],
        "lose": ["defeat", "loss", "fail"],
        "kill": ["eliminate", "frag", "take out"],
        "die": ["perish", "get killed", "fall"],
        "damage": ["hurt", "harm", "hit"],
        "heal": ["restore", "recover", "regenerate"],
        "fast": ["quick", "rapid", "speedy"],
        "slow": ["sluggish", "laggy"],
        "strong": ["powerful", "mighty", "op"],
        "weak": ["feeble", "underpowered", "trash tier"],
        "noob": ["newbie", "beginner", "rookie"],
        "pro": ["expert", "veteran", "skilled"],
        
        # Intensifiers
        "very": ["really", "super", "extremely", "so"],
        "really": ["very", "truly", "genuinely"],
        "so": ["very", "really", "extremely"],
        "always": ["constantly", "forever", "endlessly"],
        "never": ["not ever", "at no time"],
    }
    
    def __init__(
        self,
        p_synonym: float = 0.1,
        p_delete: float = 0.1,
        p_swap: float = 0.1,
        p_insert: float = 0.1,
        p_typo: float = 0.05,
        min_words: int = 3,
    ):
        """
        Initialize augmenter.
        
        Args:
            p_synonym: Probability of synonym replacement per word
            p_delete: Probability of random deletion per word
            p_swap: Probability of random swap per word pair
            p_insert: Probability of random insertion per word
            p_typo: Probability of character typo per word
            min_words: Minimum words to keep after augmentation
        """
        self.p_synonym = p_synonym
        self.p_delete = p_delete
        self.p_swap = p_swap
        self.p_insert = p_insert
        self.p_typo = p_typo
        self.min_words = min_words
        
        # Keyboard layout for typos
        self.keyboard_neighbors = {
            'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfcxs',
            'e': 'rdsw', 'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg',
            'i': 'uojk', 'j': 'uikmnh', 'k': 'iolmj', 'l': 'opk',
            'm': 'njk', 'n': 'bhjm', 'o': 'iplk', 'p': 'ol',
            'q': 'wa', 'r': 'edft', 's': 'wedxza', 't': 'rfgy',
            'u': 'yihj', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
            'y': 'tghu', 'z': 'asx'
        }
    
    def augment(self, text: str, n_aug: int = 1) -> List[str]:
        """
        Generate augmented versions of text.
        
        Args:
            text: Input text
            n_aug: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        augmented = []
        
        for _ in range(n_aug):
            words = text.split()
            
            if len(words) < self.min_words:
                augmented.append(text)
                continue
            
            # Apply augmentations
            aug_words = words.copy()
            
            # Random choice of augmentation strategy
            strategy = random.choice(['synonym', 'delete', 'swap', 'insert', 'typo'])
            
            if strategy == 'synonym':
                aug_words = self._synonym_replacement(aug_words)
            elif strategy == 'delete':
                aug_words = self._random_deletion(aug_words)
            elif strategy == 'swap':
                aug_words = self._random_swap(aug_words)
            elif strategy == 'insert':
                aug_words = self._random_insertion(aug_words)
            elif strategy == 'typo':
                aug_words = self._add_typos(aug_words)
            
            if len(aug_words) >= self.min_words:
                augmented.append(' '.join(aug_words))
            else:
                augmented.append(text)
        
        return augmented
    
    def _synonym_replacement(self, words: List[str]) -> List[str]:
        """Replace words with synonyms."""
        result = []
        for word in words:
            if random.random() < self.p_synonym:
                lower_word = word.lower()
                if lower_word in self.SYNONYMS:
                    synonym = random.choice(self.SYNONYMS[lower_word])
                    # Preserve case
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                    result.append(synonym)
                else:
                    result.append(word)
            else:
                result.append(word)
        return result
    
    def _random_deletion(self, words: List[str]) -> List[str]:
        """Randomly delete words."""
        if len(words) <= self.min_words:
            return words
        
        result = []
        for word in words:
            if random.random() >= self.p_delete:
                result.append(word)
        
        # Ensure minimum words
        if len(result) < self.min_words:
            return words
        
        return result
    
    def _random_swap(self, words: List[str]) -> List[str]:
        """Randomly swap adjacent words."""
        result = words.copy()
        for i in range(len(result) - 1):
            if random.random() < self.p_swap:
                result[i], result[i + 1] = result[i + 1], result[i]
        return result
    
    def _random_insertion(self, words: List[str]) -> List[str]:
        """Randomly insert synonyms of existing words."""
        result = []
        for word in words:
            result.append(word)
            if random.random() < self.p_insert:
                lower_word = word.lower()
                if lower_word in self.SYNONYMS:
                    synonym = random.choice(self.SYNONYMS[lower_word])
                    result.append(synonym)
        return result
    
    def _add_typos(self, words: List[str]) -> List[str]:
        """Add realistic typos."""
        result = []
        for word in words:
            if random.random() < self.p_typo and len(word) > 2:
                typo_type = random.choice(['swap', 'delete', 'neighbor', 'duplicate'])
                word = self._apply_typo(word, typo_type)
            result.append(word)
        return result
    
    def _apply_typo(self, word: str, typo_type: str) -> str:
        """Apply a specific typo to a word."""
        if len(word) < 2:
            return word
        
        idx = random.randint(0, len(word) - 1)
        
        if typo_type == 'swap' and idx < len(word) - 1:
            # Swap adjacent characters
            chars = list(word)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            return ''.join(chars)
        
        elif typo_type == 'delete':
            # Delete a character
            return word[:idx] + word[idx + 1:]
        
        elif typo_type == 'neighbor':
            # Replace with keyboard neighbor
            char = word[idx].lower()
            if char in self.keyboard_neighbors:
                neighbor = random.choice(self.keyboard_neighbors[char])
                return word[:idx] + neighbor + word[idx + 1:]
        
        elif typo_type == 'duplicate':
            # Duplicate a character
            return word[:idx] + word[idx] + word[idx:]
        
        return word


class MixUpAugmenter:
    """
    MixUp augmentation for text classification.
    
    Implements MixUp at the embedding level by interpolating
    hidden representations of two samples.
    
    Reference: Zhang et al. (2017) "mixup: Beyond Empirical Risk Minimization"
    """
    
    def __init__(self, alpha: float = 0.4):
        """
        Initialize MixUp augmenter.
        
        Args:
            alpha: Beta distribution parameter for mixing coefficient
        """
        self.alpha = alpha
    
    def get_lambda(self) -> float:
        """
        Get mixing coefficient from Beta distribution.
        
        Returns:
            Lambda value for interpolation
        """
        if self.alpha > 0:
            lam = random.betavariate(self.alpha, self.alpha)
            return max(lam, 1 - lam)  # Ensure lambda >= 0.5
        return 1.0
    
    def mix_embeddings(
        self,
        embeddings1,
        embeddings2,
        labels1,
        labels2,
        lam: Optional[float] = None,
    ) -> Tuple:
        """
        Mix two sets of embeddings.
        
        Args:
            embeddings1: First batch of embeddings
            embeddings2: Second batch of embeddings
            labels1: First batch of labels
            labels2: Second batch of labels
            lam: Optional mixing coefficient
            
        Returns:
            Tuple of (mixed_embeddings, mixed_labels)
        """
        if lam is None:
            lam = self.get_lambda()
        
        mixed_embeddings = lam * embeddings1 + (1 - lam) * embeddings2
        mixed_labels = lam * labels1 + (1 - lam) * labels2
        
        return mixed_embeddings, mixed_labels


def balance_with_augmentation(
    texts: List[str],
    labels,
    target_ratio: float = 0.5,
    augmenter: Optional[TextAugmenter] = None,
) -> Tuple[List[str], List]:
    """
    Balance dataset by augmenting minority class samples.
    
    Args:
        texts: List of text samples
        labels: Array of labels (n_samples, n_labels)
        target_ratio: Target ratio of positive to total samples per label
        augmenter: Text augmenter instance
        
    Returns:
        Augmented texts and labels
    """
    import numpy as np
    
    if augmenter is None:
        augmenter = TextAugmenter()
    
    labels = np.array(labels)
    n_samples, n_labels = labels.shape
    
    augmented_texts = list(texts)
    augmented_labels = list(labels)
    
    for label_idx in range(n_labels):
        # Find positive samples for this label
        positive_mask = labels[:, label_idx] == 1
        negative_mask = ~positive_mask
        
        n_positive = positive_mask.sum()
        n_negative = negative_mask.sum()
        
        if n_positive == 0:
            continue
        
        # Calculate how many augmented samples needed
        current_ratio = n_positive / (n_positive + n_negative)
        if current_ratio >= target_ratio:
            continue
        
        n_needed = int((target_ratio * n_negative) / (1 - target_ratio) - n_positive)
        n_needed = min(n_needed, n_positive * 5)  # Cap at 5x augmentation
        
        if n_needed <= 0:
            continue
        
        # Get positive samples
        positive_indices = np.where(positive_mask)[0]
        
        # Augment
        for _ in range(n_needed):
            idx = random.choice(positive_indices)
            aug_texts = augmenter.augment(texts[idx], n_aug=1)
            if aug_texts:
                augmented_texts.append(aug_texts[0])
                augmented_labels.append(labels[idx])
    
    return augmented_texts, np.array(augmented_labels)


if __name__ == "__main__":
    # Test augmentation
    augmenter = TextAugmenter()
    
    test_texts = [
        "You are such a bad player in this game",
        "This team is really trash and I hate them",
        "Great game everyone, that was fun!",
        "Stop being so annoying and play better noob",
    ]
    
    print("Text Augmentation Test")
    print("=" * 60)
    
    for text in test_texts:
        print(f"Original: {text}")
        augmented = augmenter.augment(text, n_aug=3)
        for i, aug in enumerate(augmented, 1):
            print(f"  Aug {i}: {aug}")
        print("-" * 60)
