import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# NLTK-ku thevaiyaana resources-ah download panrom
nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text, num_sentences=2):
    # 1. Tokenization (Sentences and Words-ah pirikkurom)
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    
    # 2. Word Frequency Table (Varthaigalin count)
    freq_table = {}
    for word in words:
        if word not in stop_words and word.isalnum():
            freq_table[word] = freq_table.get(word, 0) + 1
            
    # 3. Sentence Scoring (Ovvoru sentence-kum mark podurom)
    sentences = sent_tokenize(text)
    sentence_scores = {}
    
    for sentence in sentences:
        for word, freq in freq_table.items():
            if word in sentence.lower():
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq
                
    # 4. Get Top Sentences
    import heapq
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

# Let's Test it!
article = """
Artificial intelligence is a field of computer science that aims to create intelligent machines. 
It has become an essential part of the technology industry. 
Research associated with artificial intelligence is highly technical and specialized. 
The core problems of artificial intelligence include programming computers for certain traits such as knowledge, reasoning, problem-solving, perception, learning, and planning. 
Machine learning is also a core part of AI, where computers learn from data.
"""

print("Original Text Length:", len(article.split()))
summary = summarize_text(article)
print("\n--- AI Summary ---")
print(summary)
print("\nSummary Length:", len(summary.split()))