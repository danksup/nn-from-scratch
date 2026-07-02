#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "tokenizer.h"

typedef struct {
    char **array;
    int count;
    int capacity;
} WordList;

typedef struct {
    char *first;
    char *second;
    int frequency;
}Pair;

typedef struct {
    Pair *pairs;
    int count;
    int capacity;
}PairList;

typedef struct {
    WordList *words;
    int count;
}TokenCorpus;

typedef struct {
    char *first;
    char *second;
    char *merged;
    int rank;
} MergeRule;

typedef struct {
    MergeRule *rules;
    int count;
    int capacity;
} MergeList;

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

void add_pair(PairList *pairlist, Pair pair){
    pair.frequency = 1;
    if (pairlist->count >= pairlist->capacity){
        pairlist->capacity *= 2;
        pairlist->pairs = realloc(pairlist->pairs, pairlist->capacity * sizeof(Pair));
    }
    for (int i = 0; i < pairlist->count; i++){
        if (strcmp(pairlist->pairs[i].first, pair.first) == 0 && strcmp(pairlist->pairs[i].second, pair.second) == 0){
            pairlist->pairs[i].frequency++;
            return;
        }
    }
    pairlist->pairs[pairlist->count] = pair;
    pairlist->count++;

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

PairList get_pair_count(TokenCorpus tokencorpus){
    PairList pairlist;
    pairlist.capacity = 16;
    pairlist.pairs = malloc(pairlist.capacity * sizeof(Pair));
    pairlist.count = 0;

    for (int i = 0; i < tokencorpus.count; i++){
        for (int j = 0; j < tokencorpus.words[i].count - 1; j++){
            char *pair1 = tokencorpus.words[i].array[j];
            char *pair2 = tokencorpus.words[i].array[j+1];
            Pair pair;
            pair.first = pair1;
            pair.second = pair2;
            add_pair(&pairlist, pair);
        }
    } 
    return pairlist;
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
        split_word.capacity = len;
        split_word.array = malloc(len * sizeof(char *));
        split_word.count = 0;
        for (int j = 0; j < len; j++){
            char *token = malloc(2);
            token[0] = raw[j];
            token[1] = '\0';
            add_token(&split_word, token);
        }

        corpus.words[i] = split_word;
    }
    return corpus;
}

Pair best_pair(PairList *pairlist){
    Pair best = pairlist->pairs[0];
    for (int i = 1; i < pairlist->count; i++){
        if (pairlist->pairs[i].frequency > best.frequency) {
            best = pairlist->pairs[i];
        }
    }
    return best;
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

// fit(corpus, num_merges):
//     for step in range(num_merges):
//         pairlist = get_pair_count(corpus)
//         if pairlist.count == 0:
//             break
//         best = best_pair(pairlist)
//         merge_pair(corpus, best)
//         free(pairlist.pairs)

void fit(TokenCorpus *corpus, int target_vocab_size){
    for (int i = 0; i < corpus->count; i++){
        for (int j = 0; j < corpus->words[i].count; j++){
            
        }
    }

    // for (int i = 0; i < target_vocab_size; i++){
    //     PairList pairlist = get_pair_count(*corpus);
    //     if (pairlist.count == 0){
    //         break;
    //     }
    //     Pair best = best_pair(&pairlist);
    //     merge_pair(corpus, best);
    //     free(pairlist.pairs);
    //     for (int i = 0; i < corpus->count; i++){
    //         for (int j=0; j < corpus->words[i].count; j++){
    //             printf(" %s\n", corpus->words[i].array[j]);
    //         }
    //     }
    // }
}

int main(){
    int capacity = 2;
    char *hello[] = {"hello", "hello"};
    WordList str_array = word_list(hello, capacity);
    TokenCorpus corpus = split(&str_array);
    fit(&corpus, 100);
}

