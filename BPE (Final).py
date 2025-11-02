import re
from collections import Counter
from collections import defaultdict


with open("the-verdict.txt","r", encoding="utf-8") as f: 
    raw_text = f.read()

print("Total char: ", len(raw_text))
print(raw_text[:99])


#Step 1: Preprocess the text (Split corpus into words)

prepd = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
prepd = [item.strip() for item in prepd if item.strip()]
print(prepd[:30])

#Adding </w> in the end of each word

tokens = [' '.join(list(words)) + ' </w>' for words in prepd]

print("Tokens:\n", tokens[:30])


#Step 2: Frequency of each word

v = Counter(tokens)
vocab = dict(v)
print("\nVocabulary:")
for word, freq in vocab.items():
    print(word,":", freq)


all_words = sorted(set(prepd))
vocab_size = len(all_words)

print(vocab_size)


def get_count(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        chars = word.split()
        for i in range(len(chars) - 1):
            pairs[(chars[i], chars[i + 1])] += freq
    return pairs

print(pairs)


def merge_vocab(pair, vocab_in):
    vocab_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab_in:
        new_word = word.replace(bigram, replacement)
        vocab_out[new_word] = vocab_in[word]
    return vocab_out


num_merges = 500

for i in range(num_merges):
    pairs = get_count(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)
    print("Merge",(i+1),":",best_pair)

print("\nFinal Vocabulary:")
for word in vocab.keys():
    print(word)



'''# 
corpus = [
    "low",
    "lower",
    "newest",
    "widest"
]

vocab = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3
}
'''
