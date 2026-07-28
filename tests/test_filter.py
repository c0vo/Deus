import sys
import re
from pathlib import Path

# Add root directory to sys.path to allow imports when run directly
sys.path.append(str(Path(__file__).parent.parent))

from data.filters import TICKER_PATTERN, EXCLUDED_WORDS, FINANCIAL_KEYWORDS, REDDIT_KEYWORDS

def check_should_classify(full_text):
    print(f"\nTesting: '{full_text}'")
    
    # 1. Cashtags
    if re.search(r'\$[A-Z]{1,5}\b', full_text):
        print("-> Matched Cashtag!")
        return True
        
    # 2. Tickers
    words = set(TICKER_PATTERN.findall(full_text))
    potential_tickers = words - EXCLUDED_WORDS
    if potential_tickers:
        print(f"-> Matched potential tickers: {potential_tickers}")
        return True
        
    # 3. Keywords
    full_text_lower = full_text.lower()
    all_keywords = FINANCIAL_KEYWORDS | REDDIT_KEYWORDS
    for keyword in all_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', full_text_lower):
            print(f"-> Matched keyword: {keyword}")
            return True
            
    print("-> NOISE (Filtered out)")
    return False

if __name__ == "__main__":
    check_should_classify("US President meets CEO of AI company to discuss new rules")
    check_should_classify("I went to the store today and bought some apples. It was a good day.")
    check_should_classify("Check out this new API for building websites, it is very cool.")

