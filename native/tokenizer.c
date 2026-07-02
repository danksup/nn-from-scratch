#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "tokenizer.h"

typedef struct {
    char **array;
    int count;
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

WordList word_list(char **words, int capacity){
    WordList list;
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
    words->array[words->count] = token;
    words->count++;
}

PairList get_pair_count(TokenCorpus tokencorpus){
    PairList pairlist;
    pairlist.capacity = 999;
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
                int len1 = strlen(first);
                int len2 = strlen(second);
                char *merged = malloc(len1 + len2 + 1);
                strcpy(merged, first);
                strcat(merged, second);
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

int main(){
    int capacity = 2;
    char *hello[] = {"hello", "hello"};
    WordList str_array = word_list(hello, capacity);
    TokenCorpus corpus = split(&str_array);
    PairList pairlist = get_pair_count(corpus);
    for (int i = 0; i < pairlist.count; i++){
        printf("Pair: (%s, %s) -> Freq: %d\n", 
               pairlist.pairs[i].first, 
               pairlist.pairs[i].second, 
               pairlist.pairs[i].frequency);
    }
    Pair best = best_pair(&pairlist);
    printf("%s,%s",best.first, best.second);
    free(str_array.array);
    return 0;
}

