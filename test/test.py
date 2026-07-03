from collections import Counter

def get_pairs(word:list[str]|tuple[str]) -> list[tuple[str]]:
    pairs = []
    for i in range(len(word) - 1):
        pair = (word[i], word[i+1])
        pairs.append(pair)
    return pairs
    
def build_pair_index(word_counts:dict) -> tuple[Counter, dict]:
    counts = Counter()
    pair_to_word = {}
    for word, count in word_counts.items():
        for pair in get_pairs(word):
            counts[pair] += count
            if pair not in pair_to_word:
                pair_to_word[pair] = [word]
            else:
                if word not in pair_to_word[pair]:
                    pair_to_word[pair].append(word)
    
    return counts, pair_to_word

a = {("h","e","l","l","o"):3, ("h", "e", "r","o"):1}

b,c = build_pair_index(a)
print(c)
