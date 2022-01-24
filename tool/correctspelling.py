commas = ['.', ',', '!', '...', '......', '?', '"', '\'']



def correct(words, type=0):
    from spellchecker import SpellChecker
    spell = SpellChecker()
    # 是字符，也返回字符
    if type == 0:
        words = words.split(' ')
        for i in range(len(words)):
            cur_word = spell.correction(words[i])
            if words[i] != cur_word:
                if words[i][-1] in commas:
                    if words[i][:-1] != spell.correction(words[i][:-1]):
                        words[i] = spell.correction(words[i][:-1]) + words[i][-1]
                    else:
                        continue
                else:
                    words[i] = cur_word
        return ' '.join(words)

    # 是列表，也返回列表
    elif type == 1:
        for i in range(len(words)):
            cur_word = spell.correction(words[i])
            if words[i] != cur_word:
                if words[i][-1] in commas:
                    if words[i][:-1] != spell.correction(words[i][:-1]):
                        words[i] = spell.correction(words[i][:-1]) + words[i][-1]
                        print("===", words[i])
                    else:
                        continue
                else:
                    words[i] = cur_word
        return words

    # 类型错误
    else:
        print("type error")
        return


if __name__ == '__main__':
    # 字符串测试
    words = "If the studant asks both his friends and his doctoars, he is able to use his judgement skills to determina which choice will be best for him in the long run."
    print(words)
    words2 = correct(words)
    print(words2)

    # 列表测试
    words = "If the studant asks both his friends and his doctoars, he is able to use his judgement skills to determina which choice will be best for him in the long run."
    words = words.split(' ')
    print(words)
    words2 = correct(words, 1)
    print(words2)