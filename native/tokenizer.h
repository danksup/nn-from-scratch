#ifndef TOKENIZER_H
#define TOKENIZER_H

typedef struct {
    char **array;
    int count;
    int capacity;
} WordList;

typedef struct {
    char *first;
    char *second;
    int frequency;
} Pair;

typedef struct {
    Pair *pairs;
    int count;
    int capacity;
} PairList;

typedef struct {
    WordList *words;
    int count;
} TokenCorpus;

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

typedef struct {
    char *token;
    int id;
} VocabEntry;

typedef struct {
    VocabEntry *entries; 
    int count;
    int capacity;
} Vocab;

typedef struct {
    Vocab vocab;
    MergeList mergelist;
} Tokenizer;



WordList word_list(char **words, int capacity);
void add_pair(PairList *pairlist, Pair pair);
void add_token(WordList *words, char* token);
void add_merge_rule(MergeList *mergelist, MergeRule mergerule);
int add_vocab(Vocab *vocab, char* token);

PairList get_pair_count(TokenCorpus tokencorpus);
TokenCorpus split(WordList *words);
Pair best_pair(PairList *pairlist);
char *merge(char *first, char *second);
void merge_pair(TokenCorpus *corpus, Pair best_pair);

Tokenizer fit(TokenCorpus *corpus, int target_vocab_size);
Tokenizer fit_from_words(char **words, int word_count, int target_vocab_size);

#endif // TOKENIZER_H