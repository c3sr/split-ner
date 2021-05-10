    @staticmethod
    ## YJ : change the return value to avoid 0
    def handle_punctuation(word, punctuation_type):
        all_punctuations = list(",;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        if punctuation_type == "type1":
            #return 1 if word in all_punctuations else 0
            return 1 if word in all_punctuations else 2
        if punctuation_type == "type1-and":
            if word in all_punctuations:
                return 1 #0
            if word.lower() in ["and"]:
                return 2 #1
            return 3 #-1
        if punctuation_type == "type2":
            punctuation_vocab = list(".,-/()")
            if word in punctuation_vocab:
                return punctuation_vocab.index(word)+1  #punctuation_vocab.index(word)
            if word in all_punctuations:
                # catch all other punctuations (P)
                return len(punctuation_vocab)+1 #len(punctuation_vocab)
            return len(punctuation_vocab)+2 #0  # non-punctuation (O)
        raise NotImplementedError

