import numpy as np
from utils import *


def create_vocab_file(train_path, vocab_path, unk_threshold=3, UNK='<UNK>'):
    with open(vocab_path, 'w') as f:
        f.write("")
        f.close()
        
    train_file = open(train_path)
    
    tags = []
    tag_to_index = {}
    index_to_tag = {}
    
    words_counts = []
    words = []
    word_to_index = {}
    index_to_word = {}
    
    counter = 0
    n_lines = get_num_lines(train_path)
    
    while True:
        line = train_file.readline()
        
        if counter % 10000 == 0:
            print(str(counter) + ' / ' + str(n_lines), end='\r')
        
        counter += 1
        
        if not line:
            break
        
        if len(line.strip()) == 0:
            continue
    
        word, tag = line.strip().split()
        
        if tag not in tags:
            tag_index = len(tags)
            tag_to_index[tag] = tag_index
            index_to_tag[tag_index] = tag
            tags.append(tag)
        else:
            tag_index = tag_to_index[tag]
            
        if word not in words:
            word_index = len(words)
            words.append(word)
            words_counts.append(0)
            word_to_index[word] = word_index
            index_to_word[word_index] = word
        else:
            word_index = word_to_index[word]
            words_counts[word_index] += 1

    train_file.close()
                    
    if unk_threshold is None:
        unk_threshold = np.round(np.mean(words_counts)/2)

    new_word_to_index = {UNK: 0}
    new_index_to_word = {0: UNK}
    
    with open(vocab_path, 'a+') as f:
        line = '<UNK>' + '\n'
        f.write(line)
        f.close()
    
    with open(vocab_path, 'w') as f:
        for i, w in enumerate(words):
            c = words_counts[i]
            if c >= unk_threshold:
                word_index = len(new_word_to_index.keys())
                new_word_to_index[w] = word_index
                new_index_to_word[word_index] = w
                line = w + '\n'
                f.write(line)

    with open(vocab_path[:-3]+'tag', 'w') as f:    
        for i, s in enumerate(tags):
            line = s + '\n'
            f.write(line)
                
    return new_word_to_index, new_index_to_word, tag_to_index, index_to_tag


def calculate_model_parameters(train_path, word_to_index, index_to_word, tag_to_index, index_to_tag):
    """iterates through the training file and counts bigram and other things required to estimate model parameters

    Args:
        train_path (string): path to training file

    Returns:
        tuple: A, B, Pi, word_to_index, index_to_word, tag_to_index, index_to_tag
    """
    train_file = open(train_path)
    
    sequence_start = True
    
    words = word_to_index.keys()
    
    n_states = len(tag_to_index.keys())
    n_words = len(words)
        
    tag_bigrams = np.zeros((n_states, n_states))
    p_tag_to_word= np.zeros((n_states, n_words))
    p_tag_start = np.zeros((n_states,))
    
    counter = 0
    n_lines = get_num_lines(train_path)
    
    while True:
        line = train_file.readline()
        
        if counter % 10000 == 0:
            print(str(counter) + ' / ' + str(n_lines), end='\r')
        
        counter += 1
        
        if not line:
            break
        
        if len(line.strip()) == 0:
            sequence_start = True
            continue
    
        word, tag = line.strip().split()
        
        tag_index = tag_to_index[tag]
                
        if sequence_start:
            p_tag_start[tag_index] += 1
        else:
            tag_bigrams[prev_index, tag_index] += 1
        
        if word not in words:
            word_index = 0
        else:
            word_index = word_to_index[word]
            
        p_tag_to_word[tag_index, word_index] += 1
        
        prev_index = tag_index
        
        sequence_start = False
        
    train_file.close()
    smooth = 1
    return (smooth+tag_bigrams) / np.sum(smooth+tag_bigrams, axis=1, keepdims=True), ((p_tag_to_word+smooth) / np.sum(smooth+p_tag_to_word, axis=1, keepdims=True)).T, (smooth+p_tag_start) / np.sum(smooth+p_tag_start)

                
def evaluate_PoS(test_path, result_path, A, B, Pi, word_to_index, index_to_word, tag_to_index, index_to_tag, UNK='<UNK>'):
    """using model parameters, predicts PoS tags in the test files and evaluates them using actual labels

    Args:
        test_path (string): path to test file
        result_path (string): path to result file to write the results
        A (2D array): transition matrix
        B (2D array): observation matrix
        Pi (1D array): starting state vector
        word_to_index (dict): dictionary to convert words to corresponding indices
        index_to_word (dict): dictionary to convert words to corresponding tags
        tag_to_index (dict): dictionary to convert tags to corresponding indices
        index_to_tag (dict): dictionary to convert indices to corresponding tags
    """
    test_file = open(test_path)
    
    counter = 0
    n_lines = get_num_lines(test_path)
    
    sequence = []
    true_PoS = []
    
    confusion_matrix = np.zeros_like(A)
    
    while True:
        line = test_file.readline()

        if counter % 1000 == 0:
            print(str(counter) + ' / ' + str(n_lines), end='\r')
        
        counter += 1
        
        if not line:
            break
        
        if len(line.strip()) == 0:
            tagged_PoS = PoS_tag(sequence, A, B, Pi, word_to_index, index_to_tag)
            write_to_result(sequence, true_PoS, tagged_PoS, result_path, index_to_tag, index_to_word)
            update_confusion_matrix(confusion_matrix, tagged_PoS, true_PoS, tag_to_index)
            sequence = []
            true_PoS = []
        else:
            word, tag = line.strip().split()
            if word not in word_to_index.keys():
                word = UNK

            sequence.append(word_to_index[word])
            true_PoS.append(tag_to_index[tag])
            
    test_file.close()
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix) * 100, confusion_matrix
    
    
def update_confusion_matrix(confusion_matrix, tagged_PoS, true_PoS, tag_to_index):
    """updates confusion matrix in respect to new sequence in place (passed with reference)

    Args:
        confusion_matrix (2D array): confusion_matrix[i, j] represents number of words with actual tag i and predicted tag j
        tagged_PoS (list): indices of estimated tags for the sequence
        true_PoS (lisr): indices of true tags for the sequence
    """
    for i in range(len(true_PoS)):
        confusion_matrix[true_PoS[i], tag_to_index[tagged_PoS[i]]] += 1
        

def write_to_result(sequence, true_PoS, tagged_PoS, result_path, index_to_tag, index_to_word):
    """writes results to a file
    for each sequence 3 lines will be writter.
    firet line is the original sentence.
    second line is the true tag.
    third line is the estimated tag.
    and fouth line is empty.

    Args:
        sequence (list): indeices of words in the sequence
        true_PoS (lisr): indices of true  tags for the sequence
        tagged_PoS (list): indices of estimated tags for the sequence
        result_path (string): path to result file to write the results
        index_to_tag (dict): dictionary to convert indices to corresponding tags
        index_to_word (dict): dictionary to convert words to corresponding tags
    """
    mattrix = [
        [index_to_word[w] for w in sequence],
        [index_to_tag[t] for t in true_PoS],
        tagged_PoS
    ]
    s = [[str(e) for e in row] for row in mattrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    
    result_file = open(result_path, 'a+')
    result_file.write('\n'.join(table))
    result_file.write('\n\n')
    result_file.close()


def PoS_tag(sequence, A, B, Pi, word_to_index, index_to_tag):
    iS = list(range(len(Pi)))
    q = viterbi(iS, A, B, Pi, sequence, word_to_index)
    return [index_to_tag[t] for t in q]


def compute_delta_tilde(j, t, iS, A_tilde, B_tilde, Pi_tilde, O, delta_tilde, psi, v_to_k):
    """Calculates alpha 

    Args:
        j (int): current state index
        t (int): current time
        iS (list of ints): a list containing all state ineices
        A (2D array): transition matrix (A[s1, s2] represents P(s2 | s1) which is probability of transition from s1 to s2)
        B (2D array): observation matrix (B[k, s]] represents P(o | s) which is probability of observing o when being in state s and k is index of o in V(vocabulary list))
        Pi (1D array): initial state (Pi[s] represents P(s | <START>))
        O (string): observations (O[t] is Observation at time t)
        alpha (2D array): contains computed alpha values (non computed values are filled with None)
        v_to_k (dict): converts each word to its corresponding index

    Returns:
        float: alpha(s, t) which is probability of being in state s on time t.
    """
    if t >= len(O):
        for i in iS:
            compute_delta_tilde(i, t-1, iS, A_tilde, B_tilde, Pi_tilde, O, delta_tilde, psi, v_to_k)
        return
    
    if delta_tilde[t, j] is not None:
        return delta_tilde[t, j]
    
    if t == 0:
        delta_tilde[t, j] = Pi_tilde[j] + B_tilde[O[t], j]
        psi[t, j] = 0
        return delta_tilde[t, j]

    max_delta_tilde_t_j = -np.inf
    for i in iS:
        delta_tilde_t_j = compute_delta_tilde(i, t-1, iS, A_tilde, B_tilde, Pi_tilde, O, delta_tilde, psi, v_to_k) + A_tilde[i, j]
        if delta_tilde_t_j > max_delta_tilde_t_j:
            max_delta_tilde_t_j = delta_tilde_t_j
            arg_max = i
    
    psi[t, j] = arg_max
        
    delta_tilde[t, j] = delta_tilde_t_j + B_tilde[O[t], j]
    return delta_tilde[t, j]


def get_delta_tilde(iS, A, B, Pi, O, v_to_k):
    """calculates and return a 2D array containing all alphas: alpha[t, i] is slpha at time t and state i

    Args:
        iS (list of ints): a list containing all state ineices
        A (2D array): transition matrix (A[s1, s2] represents P(s2 | s1) which is probability of transition from s1 to s2)
        B (2D array): observation matrix (B[s, k]] represents P(o | s) which is probability of observing o when being in state s and k is index of o in V(vocabulary list))
        Pi (1D array): initial state (Pi[s] represents P(s | <START>))
        O (string): observations (O[t] is Observation at time t)
        v_to_k (dict): converts each word to its corresponding index

    Returns:
        dict: alpha
    """
    delta_tilde = np.full((len(O), len(iS)), None)
    psi = np.full((len(O), len(iS)), None)
    compute_delta_tilde(0, 1+len(O), iS, np.log(A), np.log(B), np.log(Pi), O, delta_tilde, psi, v_to_k), 
    return delta_tilde.astype(float), psi.astype(int)

def compute_delta(j, t, iS, A_tilde, B_tilde, Pi_tilde, O, delta_tilde, psi, v_to_k):
    """Calculates alpha 

    Args:
        j (int): current state index
        t (int): current time
        iS (list of ints): a list containing all state ineices
        A (2D array): transition matrix (A[s1, s2] represents P(s2 | s1) which is probability of transition from s1 to s2)
        B (2D array): observation matrix (B[k, s]] represents P(o | s) which is probability of observing o when being in state s and k is index of o in V(vocabulary list))
        Pi (1D array): initial state (Pi[s] represents P(s | <START>))
        O (string): observations (O[t] is Observation at time t)
        alpha (2D array): contains computed alpha values (non computed values are filled with None)
        v_to_k (dict): converts each word to its corresponding index

    Returns:
        float: alpha(s, t) which is probability of being in state s on time t.
    """
    if t >= len(O):
        for i in iS:
            compute_delta(i, t-1, iS, A_tilde, B_tilde, Pi_tilde, O, delta_tilde, psi, v_to_k)
        return
    
    if delta_tilde[t, j] is not None:
        return delta_tilde[t, j]
    
    if t == 0:
        delta_tilde[t, j] = Pi_tilde[j] * B_tilde[O[t], j]
        psi[t, j] = 0
        return delta_tilde[t, j]

    max_delta_tilde_t_j = 0
    arg_max = 0
    for i in iS:
        delta_tilde_t_j = compute_delta(i, t-1, iS, A_tilde, B_tilde, Pi_tilde, O, delta_tilde, psi, v_to_k) * A_tilde[i, j]
        if delta_tilde_t_j > max_delta_tilde_t_j:
            max_delta_tilde_t_j = delta_tilde_t_j
            arg_max = i
    
    psi[t, j] = arg_max
        
    delta_tilde[t, j] = delta_tilde_t_j * B_tilde[O[t], j]
    return delta_tilde[t, j]


def get_delta(iS, A, B, Pi, O, v_to_k):
    """calculates and return a 2D array containing all alphas: alpha[t, i] is slpha at time t and state i

    Args:
        iS (list of ints): a list containing all state ineices
        A (2D array): transition matrix (A[s1, s2] represents P(s2 | s1) which is probability of transition from s1 to s2)
        B (2D array): observation matrix (B[s, k]] represents P(o | s) which is probability of observing o when being in state s and k is index of o in V(vocabulary list))
        Pi (1D array): initial state (Pi[s] represents P(s | <START>))
        O (string): observations (O[t] is Observation at time t)
        v_to_k (dict): converts each word to its corresponding index

    Returns:
        dict: alpha
    """
    delta_tilde = np.full((len(O), len(iS)), None)
    psi = np.full((len(O), len(iS)), None)
    compute_delta(0, 1+len(O), iS, A, B, Pi, O, delta_tilde, psi, v_to_k), 
    return delta_tilde.astype(float), psi.astype(int)


def viterbi(iS, A, B, Pi, O, v_to_k, log=True):
    if log:
        delta, psi = get_delta_tilde(iS, A, B, Pi, O, v_to_k)
    else:
        delta, psi = get_delta(iS, A, B, Pi, O, v_to_k)
    q = np.zeros(len(O))
    q[-1] = np.argmax(delta[-1, ...])
    for t in range(len(q) - 2, -1, -1):
        q[t] = psi[t+1, int(q[t+1])]
    return q
    
    


def test():
    """A function to test if delta_tilde is calculated correctly using the example in lectures.

    Returns:
        dict: all delta_tildes
    """
    Pi = np.array([1, 0])
    S = ['s0', 's1']
    iS = list(range(len(S)))
    V = ['A', 'B']
    
    v_to_k = {v:k for k,v in enumerate(V)}
    
    A = np.array([[.6, .4],
                  [0,   1]])

    B = np.array([[.8, .3],
                  [.2, 0.7]])
    
    O = 'AAB'

    q = viterbi(iS, A, B, Pi, O, v_to_k)
    return q

if __name__ == '__main__':
    test()