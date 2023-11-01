from utils.function import add


def count_word(s, word):
    assert type(s) == str
    assert type(word) == str and len(word) == 1

    count = 0
    for i in range(len(s)):
        if s[i] == word:
            count = add(count, 1)
    return count
