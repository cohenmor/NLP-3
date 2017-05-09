from collections import defaultdict

from data import *
import numpy as np
import time


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags
    """
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = {}, {}, {}, {}, {}
    for sent in sents:
        sent = [("<s>", "*"), ("<s>", "*")] + sent + [("</s>", "STOP")]
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


def calc_transition_prob(prevprev, prev, curr, q_tri_counts, q_bi_counts, q_uni_counts, total_tokens, lambda1, lambda2):
    lambda3 = 1 - lambda1 - lambda2
    if ((prev, prevprev) in q_bi_counts) and ((curr, prev, prevprev) in q_tri_counts):
        tri_prob = float(q_tri_counts[(curr, prev, prevprev)]) / q_bi_counts[(prev, prevprev)]
    else:
        tri_prob = 0

    if ((curr, prev) in q_bi_counts) and (prev in q_uni_counts):
        bi_prob = float(q_bi_counts[(curr, prev)]) / q_uni_counts[prev]
    else:
        bi_prob = 0

    uni_prob = float(q_uni_counts.get(curr, 0)) / total_tokens

    final_prob = lambda1 * tri_prob + lambda2 * bi_prob + lambda3 * uni_prob
    return None if final_prob == 0 else np.log(final_prob)


def calc_emission_prob(word, tag, e_word_tag_counts, e_tag_counts):
    word_cnt = e_word_tag_counts[tag].get(word, 0)
    tag_cnt = e_tag_counts[tag]
    final_prob = float(word_cnt) / tag_cnt
    return None if final_prob == 0 else np.log(final_prob)


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts, lambda1,
                lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    possible_tags = e_tag_counts.keys()
    K = 50
    best_hypotheses = defaultdict(lambda: defaultdict(dict))
    best_hypotheses[0][('*','*')] = 0
    q_cache = {}
    e_cache = {}
    for i in range(1, len(sent) + 1):
        for v in possible_tags:
            best_prob = -float("inf")
            for (w, u) in best_hypotheses[i - 1]:
                p = best_hypotheses[i - 1][(w, u)]
                if (w, u, v) not in q_cache:
                    q_cache[(w, u, v)] = calc_transition_prob(w, u, v, q_tri_counts, q_bi_counts, q_uni_counts, total_tokens, lambda1,
                                                              lambda2)
                q = q_cache[(w, u, v)]
                if (sent[i-1][0], v) not in e_cache:
                    e_cache[(sent[i-1][0], v)] = calc_emission_prob(sent[i - 1][0], v, e_word_tag_counts, e_tag_counts)
                e = e_cache[(sent[i-1][0], v)]
                if q is not None and e is not None and p + q + e > best_prob:
                    sum_log_prob = p + q + e
                    best_prob = sum_log_prob
                    if len(best_hypotheses[i]) < K:
                        best_hypotheses[i][(u, v)] = sum_log_prob
                    else:
                        min_key = min(best_hypotheses[i], key=best_hypotheses[i].get)
                        if best_hypotheses[i][min_key] < sum_log_prob:
                            best_hypotheses[i].pop(min_key)
                            best_hypotheses[i][(u, v)] = sum_log_prob

    best_prob = -float("inf")
    best_v = best_u = ""
    for (u, v) in best_hypotheses[len(sent)]:
            q = calc_transition_prob(u, v, "STOP", q_tri_counts, q_bi_counts, q_uni_counts, total_tokens, lambda1, lambda2)
            p = best_hypotheses[len(sent)][(u, v)]
            prob = p + q
            if q is not None and prob > best_prob:
                best_prob = prob
                best_u = u
                best_v = v
            
    predicted_tags[len(sent) - 1] = best_v
    predicted_tags[len(sent) - 2] = best_u
    for k in range(len(sent) - 3, -1, -1):
        predicted_tags[k] = max(best_hypotheses[k + 2], key=best_hypotheses[k + 2].get)[0]
    return predicted_tags


def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts, lambda1,
             lambda2):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    total_test_tokens = 0
    sent_cnt = 0
    start = time.time()
    for sent in test_data:
        predicted_tags = hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                                     e_tag_counts, lambda1, lambda2)
        actual_tags = [token[1] for token in sent]
        for i in range(len(sent)):
            if predicted_tags[i] == actual_tags[i]:
                acc_viterbi += 1
            total_test_tokens += 1
        sent_cnt += 1
        if sent_cnt % 100 == 0:
            # Print progress reports
            end = time.time()
            print "sent cnt is: " + str(sent_cnt)
            print "curr acc is: " + str(acc_viterbi/total_test_tokens)
            print "took: " + str(end - start) + " seconds"
            start = time.time()
    acc_viterbi /= total_test_tokens
    ### END YOUR CODE
    return str(acc_viterbi)


def grid_search_lambdas(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                        e_tag_counts):
    iter_counter = 0
    opt_score = 0
    opt_lambda1 = opt_lambda2 = 0
    for lambda1 in np.arange(0.0, 1.0, 0.1):
        for lambda2 in np.arange(0.0, 1.0 - lambda1, 0.1):
            acc_viterbi = float(
                hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                         e_tag_counts, lambda1, lambda2))
            if acc_viterbi > opt_score:
                opt_score = acc_viterbi
                opt_lambda1, opt_lambda2 = lambda1, lambda2
                print("Reached new high! : " + str(acc_viterbi))

            print(str(iter_counter) + ", for lambda1 = " + str(lambda1) +
                  ", lambda2 = " + str(lambda2) +
                  ", lambda3 = " + str(round(1. - lambda1 - lambda2, 2)) +
                  ", accuracy is " + str(acc_viterbi))
            iter_counter += 1

    opt_lambda_3 = 1 - opt_lambda1 - opt_lambda2
    print "best lambda1: " + str(opt_lambda1) + \
          " best lambda2: " + str(opt_lambda2) + \
          " best lambda3: " + str(opt_lambda_3) + \
          " score: " + str(acc_viterbi)


if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                           e_tag_counts, lambda1=0.9, lambda2=0.0)  # Using optimal lambdas based on grid search

    # Uncomment next line to perform grid search
    # grid_search_lambdas(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts)
    print "dev: acc hmm viterbi: " + acc_viterbi

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                               e_word_tag_counts, e_tag_counts)
        print "test: acc hmm viterbi: " + acc_viterbi
