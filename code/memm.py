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
    features['prevprev_tag'] = prevprev_tag
    # features['prevprev_prev_tag'] = prevprev_tag + prev_tag #todo:
    # features.update(dict(("prefix_" + str(i), curr_word[:i+1]) for i in range(min(4, len(curr_word)))))
    # features.update(dict(("suffix_" + str(i), curr_word[-i-1:]) for i in range(min(4, len(curr_word)))))
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
    ### YOUR CODE HERE
    for i in range(len(sent)):
        features = extract_features(sent, i)
        vectorized_features = vectorize_features(vec, features)
        prediction = logreg.predict(vectorized_features)
        predicted_tags[i] = prediction
    ### END YOUR CODE
    return predicted_tags

def memm_viterbi(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    # possible_tags = [key for key in tagset.keys() if tagset[key] < 27] #todo: = tagset.keys()
    possible_tags = tagset.keys()
    T1 = defaultdict(lambda: defaultdict(dict))
    T2 = defaultdict(lambda: defaultdict(dict))

    # init
    T1[0]['*']['*'] = 1
    k = 1
    features = extract_features(sent, k - 1)
    vectorized_features = vectorize_features(vec, features)
    for v in possible_tags:
        T1[k]['*'][v] = 0
        q = logreg.predict_proba(vectorized_features)[0][tagset[v]]
        if q > T1[k]['*'][v]:
            T1[k]['*'][v] = q
            T2[k]['*'][v] = '*'
    k = 2
    features = extract_features(sent, k-1)
    vectorized_features = vectorize_features(vec, features)
    for u in possible_tags:
        for v in possible_tags:
            T1[k][u][v] = 0
            p = T1.get(k - 1, {}).get('*', {}).get(u, 0)
            q = logreg.predict_proba(vectorized_features)[0][tagset[v]]
            if p * q > T1[k][u][v]:
                T1[k][u][v] = p * q
                T2[k][u][v] = '*'

    for k in range(3, len(sent) + 1):
        features = extract_features(sent, k-1)
        vectorized_features = vectorize_features(vec, features)
        for u in possible_tags:
            for v in possible_tags:
                T1[k][u][v] = 0
                for t in possible_tags:
                    p = T1.get(k-1, {}).get(t, {}).get(u, 0)
                    q = logreg.predict_proba(vectorized_features)[0][tagset[v]]
                    if p * q > T1[k][u][v]:
                        T1[k][u][v] = p * q
                        T2[k][u][v] = t

    best_prob = 0
    best_v = best_u = ""
    for u in possible_tags:
        for v in possible_tags:
            if T1[len(sent)-1][u][v] > best_prob:
                best_prob = T1[len(sent)-1][u][v]
                best_v = v
                best_u = u
    predicted_tags[len(sent)-1] = best_v
    predicted_tags[len(sent) - 2] = best_u

    for k in range(len(sent)-3, -1, -1):
        predicted_tags[k] = T2[k+3][predicted_tags[k+1]][predicted_tags[k+2]]
    ### END YOUR CODE
    return predicted_tags

def memm_eval(test_data, logreg, vec):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm & greedy hmm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    token_count = 0
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
    acc_greedy /= token_count
    acc_viterbi /= token_count
    return acc_viterbi, acc_greedy

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

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

    all_examples = train_examples #todo:
    all_examples.extend(dev_examples) #todo:
    # all_examples = train_examples[:200]
    # all_examples.extend(dev_examples[:200])

    print "Vectorize examples"
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples] #todo:
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:] #todo:
    # train_examples_vectorized = all_examples_vectorized[:200]
    # dev_examples_vectorized = all_examples_vectorized[200:]
    print "Done"

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=1024, solver='lbfgs', C=100000, verbose=10, n_jobs=4) #max_iter=128
    print "Fitting..."
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print "done, " + str(end - start) + " sec"
    #End of log linear model training

    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec)
    print "dev: acc memm greedy: " + acc_greedy
    print "dev: acc memm viterbi: " + acc_viterbi
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec)
        print "test: acc memmm greedy: " + acc_greedy
        print "test: acc memmm viterbi: " + acc_viterbi