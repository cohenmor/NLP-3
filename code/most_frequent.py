from data import *

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    d = {}
    for sent in train_data:
        for tagged_pair in sent:
            word, tag = tagged_pair
            if word not in d:
                d[word] = {}
            d[word][tag] = d[word].get(tag, 0) + 1

    res = {}
    for key in d:
        res[key] = max(d[key], key=d[key].get)
    return res

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    token_count = 0.
    correct_pred_count = 0.
    for sent in test_set:
        for tagged_pair in sent:
            word, tag = tagged_pair
            if tag == pred_tags[word]:
                correct_pred_count += 1
            token_count += 1
    return str(100. * float(correct_pred_count) / token_count)

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " + most_frequent_eval(dev_sents, model)

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + most_frequent_eval(test_sents, model)