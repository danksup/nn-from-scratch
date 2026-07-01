from collections import Counter

@staticmethod

     
def get_pair_counts( words:list) -> Counter:
        counts = Counter()
        for word in words:
           for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                counts[pair] += 1
        return counts

@staticmethod
def merge_pair(words:list, best_pair:str):
    merged = []
    for word in words:
        i = 0
        n = len(word)
        new_word = []
        while i < n - 1:
            pair = word[i], word[i + 1]
            if pair == best_pair:
                new_word.append(word[i] + word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i+=1
        if i == n - 1:
            new_word.append(word[n-1])
        merged.append(new_word)
    return merged

words = ["hello", "hero"]
counter = get_pair_counts(words)
best_pair = counter.most_common(1)[0][0]
print(best_pair[0] + best_pair[1])
print(merge_pair(words, best_pair))