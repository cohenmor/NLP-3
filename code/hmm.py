from data import *
import numpy as np

def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags
    """
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = {}, {}, {}, {}, {}
    # todo: e_tag_counts ==? q_uni_counts
    for sent in sents:
        sent = [("*","*"), ("*","*")] + sent + [("STOP", "STOP")]
        for i in range(2):
            word, tag = sent[i]
            q_uni_counts[tag] = q_uni_counts.get(tag, 0) + 1
            e_tag_counts[tag] = e_tag_counts.get(tag, 0) + 1
            if tag not in e_word_tag_counts:
                e_word_tag_counts[tag] = {}
            e_word_tag_counts[tag][word] = e_word_tag_counts[tag].get(word, 0) + 1
            total_tokens += 1
        curr, prev = sent[1][1], sent[0][1]
        q_bi_counts[(curr, prev)] = q_bi_counts.get((curr, prev), 0) + 1
        for i in range(2, len(sent)):
            word, tag = sent[i]
            curr, prev, prevprev = sent[i][1], sent[i - 1][1], sent[i - 2][1]
            q_tri_counts[(curr, prev, prevprev)] = q_tri_counts.get((curr, prev, prevprev), 0) + 1
            q_bi_counts[(curr, prev)] = q_bi_counts.get((curr, prev), 0) + 1
            q_uni_counts[curr] = q_uni_counts.get(curr, 0) + 1
            e_tag_counts[tag] = e_tag_counts.get(tag, 0) + 1
            if tag not in e_word_tag_counts:
                e_word_tag_counts[tag] = {}
            e_word_tag_counts[tag][word] = e_word_tag_counts[tag].get(word, 0) + 1
            total_tokens += 1
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts

def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    T1 = np.zeros(shape=(len(e_tag_counts), len(sent)))
    T2 = np.zeros(shape=(len(e_tag_counts), len(sent)))
    states = e_tag_counts.keys()
    for i in range(len(states)):
        T1[i][0] = (q_tri_counts[(i, "*", "*")] / q_bi_counts[("*", "*")]) * \
                   (e_word_tag_counts[sent[0]][states[i]] / e_tag_counts[states[i]])
        T2[i][0] = 0
    # for k in range(2, len(sent)):
    #     for u in possible_tags:
    #         for v in possible_tags:
    #             return max()
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE
    return str(acc_viterbi)

if __name__ == "__main__":
    train_sents = read_conll_pos_file("../../Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("../../Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts)
    print "dev: acc hmm viterbi: " + acc_viterbi

    if os.path.exists("../../Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("../../Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "test: acc hmm viterbi: " + acc_viterbi