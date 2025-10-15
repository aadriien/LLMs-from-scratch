################################################################################
##  Basic Tokenizer
################################################################################

import re


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text



def read_prep_text():
    with open("chapter2/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        all_tokens = sorted(list(set(preprocessed)))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])

        vocab = {token:integer for integer,token in enumerate(all_tokens)}

        return vocab


def test_tokenizer_v1(vocab):
    tokenizer = SimpleTokenizerV1(vocab)

    text = """"It's the last he painted, you know," 
            Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)

    tokenizer.decode(ids)
    tokenizer.decode(tokenizer.encode(text))


def test_tokenizer_v2(vocab):
    tokenizer = SimpleTokenizerV2(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."

    text = " <|endoftext|> ".join((text1, text2))

    tokenizer.encode(text)
    tokenizer.decode(tokenizer.encode(text))

    print(text)



################################################################################
##  BPE Tokenizer
################################################################################

import tiktoken


def test_BPE_tokenizer():
    tokenizer = tiktoken.get_encoding("gpt2")

    # text = (
    #     "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    #     "of someunknownPlace."
    # )

    text = "Akwirw ier"

    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)

    for id in integers:
        print(f"ID: {id} -> {tokenizer.decode([id])} (token)")

    strings = tokenizer.decode(integers)
    print(strings)



################################################################################
##  MAIN    
################################################################################

if __name__ == "__main__":
    print("\nLoading outputs for basic tokenizer:\n\n")
    vocab = read_prep_text()

    test_tokenizer_v1(vocab)
    test_tokenizer_v2(vocab)

    print("\n\n################################################################################")

    print("\nLoading outputs for BPE tokenizer:\n\n")
    test_BPE_tokenizer()

