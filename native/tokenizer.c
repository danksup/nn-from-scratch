#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "tokenizer.h"

WordList word_list(char **words, int capacity){
    WordList list;
    list.capacity = capacity;
    list.count = 0;
    list.array = malloc(capacity * sizeof(char *));
    for (int i = 0; i < capacity; i++){
        
        list.array[list.count] = (char *)words[i];
        list.count++;
    }
    
    return list;
}

unsigned long hash_string(const char *str) {
    unsigned long hash = 5381;
    int c;

    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }

    return hash;
}

unsigned long hash_pair(char *first, char *second) {
    unsigned long hash = hash_string(first);

    hash = ((hash << 5) + hash) + '|';

    int c;
    while ((c = *second++)) {
        hash = ((hash << 5) + hash) + c;
    }

    return hash;
}

PairHashMap get_pair_count_hash(TokenCorpus tokencorpus){
    PairHashMap map;
    map.bucket_count = 8192;
    map.count = 0;
    map.buckets = calloc(map.bucket_count, sizeof(PairNode *));

    for (int i = 0; i < tokencorpus.count; i++){
        for (int j = 0; j < tokencorpus.words[i].count - 1; j++){
            Pair pair;
            pair.first = tokencorpus.words[i].array[j];
            pair.second = tokencorpus.words[i].array[j + 1];
            add_pair_hash(&map, pair);
        }
    }

    return map;
}

void add_pair_hash(PairHashMap *map, Pair pair) {
    unsigned long hash = hash_pair(pair.first, pair.second);
    int index = hash % map->bucket_count;

    PairNode *node = map->buckets[index];

    while (node != NULL) {
        if (strcmp(node->pair.first, pair.first) == 0 &&
            strcmp(node->pair.second, pair.second) == 0) {
            node->pair.frequency++;
            return;
        }

        node = node->next;
    }

    PairNode *new_node = malloc(sizeof(PairNode));
    pair.frequency = 1;
    new_node->pair = pair;
    new_node->next = map->buckets[index];
    map->buckets[index] = new_node;
    map->count++;
}

void add_token(WordList *words, char* token){
    if (words->count >= words->capacity){
        words->capacity *= 2;
        words->array = realloc(words->array, words->capacity * sizeof(char *));
    }
    words->array[words->count] = token;
    words->count++;
}

void add_merge_rule(MergeList *mergelist, MergeRule mergerule){
    if (mergelist->count >= mergelist->capacity){
        mergelist->capacity *= 2;
        mergelist->rules = realloc(mergelist->rules,mergelist->capacity * sizeof(MergeRule));
    }

    mergelist->rules[mergelist->count] = mergerule;
    mergelist->count++;
}

int add_vocab(Vocab *vocab, char*token){
    for (int i = 0; i < vocab->count;i++){
        if (strcmp(vocab->entries[i].token, token) == 0){
            return vocab->entries[i].id;
        }
    }

    if (vocab->count >= vocab->capacity){
        vocab->capacity *= 2;
        vocab->entries = realloc(vocab->entries, vocab->capacity * sizeof(VocabEntry));
    }

    int id = vocab->count;
    vocab->entries[id].token = strdup(token);
    vocab->entries[id].id = id;
    vocab->count++;
    return id;
}


TokenCorpus split(WordList *words){
    TokenCorpus corpus;
    corpus.count = 0;
    corpus.words = malloc(words->count * sizeof(WordList));
    corpus.count = words->count;

    for (int i = 0; i < words->count; i++){
        char *raw = words->array[i];
        int len = strlen(raw);
        WordList split_word;
        split_word.capacity = len + 1;
        split_word.array = malloc((len + 1) * sizeof(char *));
        split_word.count = 0;
        for (int j = 0; j < len; j++){
            char *token = malloc(2);
            token[0] = raw[j];
            token[1] = '\0';
            add_token(&split_word, token);
        }
        add_token(&split_word, "</w>");
        corpus.words[i] = split_word;
    }
    return corpus;
}

char *merge(char *first, char *second){
    int len1 = strlen(first);
    int len2 = strlen(second);
    char *merged = malloc(len1 + len2 + 1);
    strcpy(merged, first);
    strcat(merged, second);

    return merged;
}

void merge_pair(TokenCorpus *corpus, Pair best_pair){
    for (int i = 0; i < corpus->count; i++){
        int j = 0;
        int n = corpus->words[i].count;
        while (j < n - 1) {
            // Pair pair;
            char *first = corpus->words[i].array[j];
            // pair.first = first;
            char *second = corpus->words[i].array[j + 1];
            // pair.first = second;
            char *best_first = best_pair.first;
            char *best_second = best_pair.second;

            if (strcmp(first,best_first) == 0 && strcmp(second, best_second) == 0){
                char *merged = merge(first, second);
                corpus->words[i].array[j] = merged;
                for (int k = j+1; k < corpus->words[i].count - 1; k++){
                    corpus->words[i].array[k] = corpus->words[i].array[k+1];
                }
                corpus->words[i].count--;
                j++;
                n--;
            }
            else{
                j++;
            }
        }
    }
}

Pair best_pair_hash(PairHashMap *map) {
    Pair best;
    best.first = NULL;
    best.second = NULL;
    best.frequency = -1;

    for (int i = 0; i < map->bucket_count; i++) {
        PairNode *current = map->buckets[i];
        while (current != NULL) {
            if (current->pair.frequency > best.frequency) {
                best = current->pair;
            }
            
            PairNode *next_node = current->next;
            current = next_node;
        }
    }
    return best;
}

void free_pair_hashmap(PairHashMap *map){
    for (int i = 0; i < map->bucket_count; i++){
        PairNode *node = map->buckets[i];
        while (node != NULL){
            PairNode *next = node->next;
            free(node);
            node = next;
        }
    }

    free(map->buckets);
    map->buckets = NULL;
    map->bucket_count = 0;
    map->count = 0;

}

Tokenizer fit(TokenCorpus *corpus, int target_vocab_size){
    Vocab vocab;
    vocab.count = 0;
    vocab.capacity = 16;
    vocab.entries = malloc(vocab.capacity * sizeof(VocabEntry));

    MergeList merges;
    merges.count = 0;
    merges.capacity = 16;
    merges.rules = malloc(merges.capacity * sizeof(MergeRule));

    for (int i = 0; i < corpus->count; i++){
        for (int j = 0; j < corpus->words[i].count; j++){
            add_vocab(&vocab, corpus->words[i].array[j]);
        }
    }

    while (vocab.count < target_vocab_size) {
        PairHashMap count = get_pair_count_hash(*corpus);

        if (count.count == 0){
            free_pair_hashmap(&count);
            break;
        }

        Pair best = best_pair_hash(&count);
        char *merged_best = merge(best.first, best.second);

        MergeRule rule;
        rule.first = strdup(best.first);
        rule.second = strdup(best.second);
        rule.merged = merged_best;
        rule.rank = merges.count;

        add_merge_rule(&merges, rule);
        add_vocab(&vocab, merged_best);
        merge_pair(corpus, best);

        free_pair_hashmap(&count);
    }

    Tokenizer tokenizer;
    tokenizer.vocab = vocab;
    tokenizer.mergelist = merges;
    return tokenizer;
}

Tokenizer fit_from_words(char **words, int word_count, int target_vocab_size){
    WordList wordlist = word_list(words, word_count);
    TokenCorpus corpus = split(&wordlist);
    Tokenizer tokenizer = fit(&corpus, target_vocab_size);
    return tokenizer;
}

#ifdef TEST_TOKENIZER
int main(){
    char *hello[] = {
    "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog.",
    "Building", "a", "custom", "sub-word", "tokenizer", "from", "scratch", "requires", "careful", "memory", "alignment.",
    "Deep", "learning", "models", "rely", "on", "structured", "data", "pipelines", "to", "feed", "matrices.",
    "If", "a", "model", "has", "too", "much", "capacity", "it", "will", "memorize", "instead", "of", "generalizing.",
    "Managing", "pointers", "manually", "prevents", "silent", "data", "corruption", "and", "keeps", "execution", "boundaries", "clean.",
    "To", "break", "repetitive", "sampling", "loops", "try", "lowering", "the", "temperature", "or", "adding", "a", "penalty."
    };
    int capacity = sizeof(hello) / sizeof(hello[0]);
    WordList str_array = word_list(hello, capacity);
    TokenCorpus corpus = split(&str_array);
    Tokenizer tokenizer = fit(&corpus, 100);
    printf("=== Vocabulary ===\n");
    for (int i = 0; i < tokenizer.vocab.count; i++) {
        printf("%3d : %s\n",
            tokenizer.vocab.entries[i].id,
            tokenizer.vocab.entries[i].token);
    }

    printf("\n=== Merge Rules ===\n");
    for (int i = 0; i < tokenizer.mergelist.count; i++) {
        MergeRule rule = tokenizer.mergelist.rules[i];
        printf("%3d : (%s, %s) -> %s\n",
            rule.rank,
            rule.first,
            rule.second,
            rule.merged);
    }

    printf("\n=== Final Corpus ===\n");
    for (int i = 0; i < corpus.count; i++) {
        printf("Word %d: ", i);
        for (int j = 0; j < corpus.words[i].count; j++) {
            printf("%s ", corpus.words[i].array[j]);
        }
        printf("\n");
    }
}
#endif
