from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import numpy as np
from collections import defaultdict

def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    features['prev_word'] = prev_word
    features['next_word'] = next_word
    features['prevprev_word'] = prevprev_word
    features['prev_tag'] = prev_tag
    features['prevprev_prev_tag'] = prevprev_tag + prev_tag
    features.update(dict(("prefix_" + str(i), curr_word[:i+1]) for i in range(min(4, len(curr_word)))))
    features.update(dict(("suffix_" + str(i), curr_word[-i-1:]) for i in range(min(4, len(curr_word)))))
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents):
    print "building examples"
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tagset[sent[i][1]])
    print "done"
    return examples, labels

def memm_greeedy(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    prev_tag = prevprev_tag = '*'
    for i in range(len(sent)):
        features = extract_features(sent, i)
        features['prev_tag'] = prev_tag
        features['prevprev_prev_tag'] = prevprev_tag + prev_tag
        vectorized_features = vectorize_features(vec, features)
        prediction = logreg.predict(vectorized_features)[0]
        predicted_tags[i] = index_to_tag_dict[prediction]
        prevprev_tag = prev_tag
        prev_tag = predicted_tags[i]
    return predicted_tags

def memm_viterbi(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    possible_tags = tagset.keys()
    K = 25
    best_hypotheses = defaultdict(lambda: defaultdict(dict))

    best_hypotheses[0]['*']['*'] = 0 # v , u , w
    for i in range(1, len(sent) + 1):
        features = extract_features(sent, i - 1)
        possible_u_tags = best_hypotheses[i - 1].keys()
        # features_lst = [[] for _ in possible_u_tags]
        for u_index in range(len(possible_u_tags)):
            u = possible_u_tags[u_index]
            possible_w_tags = best_hypotheses[i - 1][u].keys()
            features_lst = [dict(features) for _ in possible_w_tags]
            for w_index in range(len(possible_w_tags)):
                w = possible_w_tags[w_index]
                features_lst[w_index]['prev_tag'] = u
                features_lst[w_index]['prevprev_prev_tag'] = w + u
            vectorized_features = vec.transform(features_lst)
            q_mat = logreg.predict_proba(vectorized_features)
            for v in possible_tags:
                # for u_index in range(len(possible_u_tags)):
                #     u = possible_u_tags[u_index]
                possible_w_tags = best_hypotheses[i - 1][u].keys()
                best_prob = -float("inf")
                for w_index in range(len(possible_w_tags)):
                    w = possible_w_tags[w_index]
                    p = best_hypotheses[i - 1][u][w]
                    q = q_mat[w_index][tagset[v]]
                    if q > 0:
                        sum_log_prob = p + np.log(q)
                        if sum_log_prob > best_prob:
                            best_hypotheses[i][v][u] = sum_log_prob
                            best_prob = sum_log_prob
                        # if len(best_hypotheses[i]) < K:
                        #     best_hypotheses[i][(u, v)] = sum_log_prob
                        # else:
                        #     min_key = min(best_hypotheses[i], key=best_hypotheses[i].get)
                        #     if best_hypotheses[i][min_key] < sum_log_prob:
                        #         best_hypotheses[i].pop(min_key)
                        #         best_hypotheses[i][(u, v)] = sum_log_prob

    best_prob = -float("inf")
    best_v = best_u = ""
    possible_v_tags = best_hypotheses[len(sent)]
    for v in best_hypotheses[len(sent)]:
        for u in best_hypotheses[len(sent)][v]:
            prob = best_hypotheses[len(sent)][v][u]
            if prob > best_prob:
                best_prob = prob
                best_u = u
                best_v = v

    predicted_tags[len(sent)-1] = best_v
    predicted_tags[len(sent) - 2] = best_u
    for k in range(len(sent) - 3, -1, -1):
        predicted_tags[k] = max(best_hypotheses[k + 2][best_u], key=best_hypotheses[k + 2][best_u].get)
        best_u = predicted_tags[k]

    ### END YOUR CODE
    return predicted_tags

def memm_eval(test_data, logreg, vec):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm & greedy hmm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    token_count = 0
    sent_cnt = 0
    start = time.time()
    for sent in test_data:
        actual = [token[1] for token in sent]
        predicted_greedy = memm_greeedy(sent, logreg, vec)
        predicted_viterbi = memm_viterbi(sent, logreg, vec)

        for i in range(len(sent)):
            if actual[i] == predicted_greedy[i]:
                acc_greedy += 1
            if actual[i] == predicted_viterbi[i]:
                acc_viterbi += 1
        token_count += len(sent)
        sent_cnt += 1
        if sent_cnt % 10 == 0:
            end = time.time()
            print "sent cnt is: " + str(sent_cnt)
            print "curr greedy acc is: " + str(acc_greedy/token_count)
            print "curr viterbi acc is: " + str(acc_viterbi/token_count)
            print "took: " + str(end-start) + " seconds"
    acc_greedy /= token_count
    acc_viterbi /= token_count
    return acc_viterbi, acc_greedy

if __name__ == "__main__":
    train_sents = read_conll_pos_file("../../Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("../../Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    #The log-linear model training.
    #NOTE: this part of the code is just a suggestion! You can change it as you wish!
    curr_tag_index = 0
    tagset = {}
    for train_sent in train_sents:
        for token in train_sent:
            tag = token[1]
            if tag not in tagset:
                tagset[tag] = curr_tag_index
                curr_tag_index += 1
    index_to_tag_dict = invert_dict(tagset)
    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print "Vectorize examples"
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=10, solver='lbfgs', C=100000, verbose=1, n_jobs=3) # todo: max_iter=128, remove n_jobs
    print "Fitting..."
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print "done, " + str(end - start) + " sec"
    #End of log linear model training

    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec)
    print "dev: acc memm greedy: " + acc_greedy
    print "dev: acc memm viterbi: " + acc_viterbi
    if os.path.exists('../../Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("../../Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec)
        print "test: acc memmm greedy: " + acc_greedy
        print "test: acc memmm viterbi: " + acc_viterbi