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

typedef struct PairNode {
    Pair pair;
    struct PairNode *next;
} PairNode;

typedef struct {
    PairNode **buckets;
    int bucket_count;
    int count;
} PairHashMap;

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
void add_token(WordList *words, char* token);
void add_merge_rule(MergeList *mergelist, MergeRule mergerule);
int add_vocab(Vocab *vocab, char* token);

TokenCorpus split(WordList *words);
char *merge(char *first, char *second);
void merge_pair(TokenCorpus *corpus, Pair best_pair);

Tokenizer fit(TokenCorpus *corpus, int target_vocab_size);
Tokenizer fit_from_words(char **words, int word_count, int target_vocab_size);

unsigned long hash_string(const char *str);
unsigned long hash_pair(char *first, char *second);
void add_pair_hash(PairHashMap *map, Pair pair);
PairHashMap get_pair_count_hash(TokenCorpus tokencorpus);
Pair best_pair_hash(PairHashMap *map);
void free_pair_hashmap(PairHashMap *map);

#endif // TOKENIZER_H