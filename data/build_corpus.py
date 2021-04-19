"""
Script to prepare the sentece data provided by the Leipzig Corpora Collection [1]
German 'Web-wrt' corpus for further processing.

Unzip the file 'deu-de_web-wrt_2019_10M.tar.gz' in the same folder as the script an run it.

The scripts removes sentence id and other metadata from the sentences and discards all 
sentences containing characters that are not letters, space, comma or end of sentence
punctuation (.!?). Output sentences are saved in 'sentences.txt'.

[1]: https://wortschatz.uni-leipzig.de/en/download/German
"""

path_web_wrt = 'deu-de_web-wrt_2019_10M/deu-de_web-wrt_2019_10M-sentences.txt'

""" Example line from the file:
'10\tAber das berücksichtigt dann eben nicht die Konsequenz, dass wir steigende Studierendenzahlen haben werden, sondern die bleiben gleich.\t14.2514\n'
"""

path_out_file = 'sentences.txt'


def valid_sentence(s: str) -> bool:
    state = True
    for c in s[:-1]:
        if c.isalpha() or c.isspace() or c == ',':
            continue
        else:
            state = False
            break
    if s[-1] not in {'.', '?', '!'}:
       state = False
    return state


def filter_sentences_from_file(in_path, out_path):
    with open(in_path, mode='r') as in_file, open(out_path, mode='a') as out_file:

         print('Processing sentences from ' + in_path.split('/')[-1] )

         lines = in_file.readlines()
         for i, line in enumerate(lines, start = 1):
             s = line.split('\t')[1].strip()
             if valid_sentence(s):
                 s = s.replace('\x85' ,'')
                 words = s[:-1].split(' ')
                 words = [word for word in words if word != '']
                 s = ' '.join(words) + s[-1]
                 out_file.write(s + '\n')
             if i % 500_000 == 0:
                print('Processed', i, 'sentences')
         print('Done!')


filter_sentences_from_file(path_web_wrt, path_out_file)
