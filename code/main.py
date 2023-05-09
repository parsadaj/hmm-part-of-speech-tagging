# to run this file, go to HW3 folder, open terminal and run: python3 code/main.py path/to/trainfile path/to/testfile path/to/save/the/vocab/file
import hmm2
import sys


def main(args):

    train_path, test_path, result_file, vocab_path = args
    
    # with open(result_file, 'w') as f:
    #     f.write("")
    #     f.close()
        
    print('------------- creating vocab -------------')
    word_to_index, index_to_word, tag_to_index, index_to_tag = hmm2.create_vocab_file(train_path, vocab_path)
    print('----------- done creating vocab ----------\n')
    
    print('---------------- training ----------------')
    A, B, Pi = hmm2.calculate_model_parameters(train_path, word_to_index, index_to_word, tag_to_index, index_to_tag)
    print('-------------- done training -------------\n')

    print('-------------- evaluating --------------\n')
    accuracy, confusion_matrix = hmm2.evaluate_PoS(test_path, result_file, A, B, Pi, word_to_index, index_to_word, tag_to_index, index_to_tag)
    print('------------ done evaluating -----------\n')

    print('accuracy: ', accuracy, '\n')
    print('confusion matrix: ')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            print(confusion_matrix[i, j], end=' ')
        print()
    
    
if __name__ == '__main__':
    pass
    main(sys.argv[1:])
    # main(('data/Train.txt', 'data/Test.txt', 'output/results.txt', 'output/vocab.txt'))