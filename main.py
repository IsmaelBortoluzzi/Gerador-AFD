import re
import string
from itertools import chain
from unicodedata import normalize

from tabulate import tabulate


def print_table(table, headers, title=None):
    if title:
        print(tabulate([title.upper()], tablefmt="fancy_grid"))
    print(tabulate(table, headers, tablefmt="fancy_grid"))


def formalize_data(file_name):
    words = list()
    grammar = list()
    with open(file_name, 'r') as file:
        for line in file:
            if line.strip() != '':
                if '::=' in line:
                   grammar.append(line.strip())
                else:
                   words.append(line.strip())

    return words, grammar


def prepare_alphabet_machine(all_tokens):
    formalize_data = [list(word) for word in all_tokens]
    return list(dict.fromkeys(list(chain.from_iterable(formalize_data))))


def get_all_tokens(words, grammar):
    all_tokens = list()
    pattern = "<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>"

    for linha in words + grammar:
        valores = normalize('NFKD', linha.strip()).encode('ASCII', 'ignore').decode('ASCII').lower()
        if valores.startswith('<s>'):
            tokens = valores.replace('::=', '').split('|')
            for token in tokens:
                all_tokens.append(re.sub(pattern, '', token).strip())
            continue
        elif valores.startswith('<'):
            continue
        all_tokens.append(valores)
    return prepare_alphabet_machine(all_tokens)


class Automata:
    def __init__(self, words, grammar):
        self.symbol_initial = 'S'
        self.sigma = '\u03B4'
        self.epsilon = '\u03B5'
        self.alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.words = words
        self.grammar = grammar
        self.alphabet_machine = get_all_tokens(self.words, self.grammar)
        self.states = [[self.symbol_initial] + [''] * len(self.alphabet_machine)]
        self.last_state = 0
        self.checked = set()

    def deep_index(self, word):
        """
            Function return index list of list (0, 0) S[0][0]
        """
        return [(i, sub.index(word)) for (i, sub) in enumerate(self.states) if word in sub]

    def get_state(self, position=0, **kwargs):
        """
            Function return actual state by autoincrement
        """
        return kwargs.get('state_final', '') + self.alphabet[self.last_state - position]

    def create_state(self, new=True, state='', **kwargs):
        """
            Function that creates the state
            state: pre-defined(in case when determize) the new state
             is the state parameter passed in function,
            state_final: passed in kwargs, generally is the '*'
        """
        if new:
            if state:
                self.states = self.states + [
                    [kwargs.get('state_final', '') + state] + [''] * len(self.alphabet_machine)
                ]
            else:
                self.states = self.states + [
                    [self.get_state(state_final=kwargs.get('state_final', ''))] + [''] * len(self.alphabet_machine)
                ]
                self.last_state += 1

    def build_afnd(self):
        """
            Function get the data and generate the automata finite
        """
        for num_word, word in enumerate(self.words):
            len_word = len(word) - 1  # decrement 1 for compare in enumerate for
            for num_token, token in enumerate(word):
                if num_token == 0:
                    # here is the first token in word or alphabet not in S
                    if num_word == 0:
                        # add in S the first token and create new state
                        self.states[0][self.alphabet_machine.index(token) + 1] += self.alphabet[self.last_state]
                        self.create_state()
                    else:
                        # only add in S the token
                        self.states[0][self.alphabet_machine.index(token) + 1] += self.alphabet[self.last_state - 1]
                else:
                    if num_token == len_word:
                        # here is the reason for -1, when it's the final token in the word add state
                        # and create new state
                        self.states[self.last_state][self.alphabet_machine.index(token) + 1] = \
                            self.alphabet[self.last_state]
                        self.create_state(state_final='*')
                        try:
                            self.words[num_word + 1]
                        except IndexError:
                            continue
                        self.create_state()
                    else:
                        # the normal process, add state and create new state
                        self.states[self.last_state][self.alphabet_machine.index(token) + 1] = \
                            self.alphabet[self.last_state]
                        self.create_state()

        self.change_states_signature()

        for sentence in self.grammar:
            state = sentence.split('::=')[0].strip()[1]
            for x in range(len(self.states)):
                if self.states[x][0] in (state, f'*{state}'):
                    tokens_with_next_state = sentence.split('::=')[1].replace(' ', '').replace('>', '').split('|')
                    self.remove_epsolon(tokens_with_next_state)
                    for token_state in tokens_with_next_state:
                        token, state = token_state.split('<')
                        self.states[x][self.alphabet_machine.index(token) + 1] += state
                    break

    def change_states_signature(self):
        for x in range(1, len(self.grammar)):
            state = self.grammar[x].split('::=')[0].strip()[1]
            self.create_state()
            if self.epsilon in self.grammar[x]:
                self.states[self.last_state][0] = '*' + self.states[self.last_state][0]

            self.replace_signature(state, self.alphabet[self.last_state - 1])

    def replace_signature(self, state, new_state):
        for x in range(len(self.grammar)):
            self.grammar[x] = self.grammar[x].replace(state, new_state)

    def remove_epsolon(self, tokens_with_next_state):
        if self.epsilon in tokens_with_next_state:
            tokens_with_next_state.remove(self.epsilon)

    def build_afd(self):
        """
            Function that determines the automata
        """
        for state in self.states:
            for signature in state:
                if not any(signature in state[0] for state in self.states) and signature:
                    state_final = False
                    # verify the token not have state, if not create the state but passed the name of state, btw the token
                    for token_new_state in signature:
                        if any('*' + token_new_state in state[0] for state in self.states) and \
                                not any('*' + signature in state[0] for state in self.states):
                            state_final = True
                    if state_final:
                        self.create_state(state=signature, state_final='*')
                    else:
                        self.create_state(state=signature)
                    indexes = self.deep_index(signature)
                    self.states[indexes[0][0]][indexes[0][1]] = self.states[indexes[0][0]][indexes[0][1]]
                    for new_token in signature:
                        # add states for new state
                        indexes_tokens = self.deep_index(new_token)
                        for key, valor in enumerate(self.states[indexes_tokens[-1][0]]):
                            if key == 0 or not valor:
                                # not subscribe the name of state
                                continue
                            self.states[-1][key] += valor

        for state in self.states:
            for signature in state:
                # its ugly but here to be recursivity that function
                if not any(signature in state[0] for state in self.states) and signature:
                    self.build_afd()

    def minimize(self):
        self.remove_unreachable()
        self.remove_deads()

    def remove_unreachable(self):
        self.checked.update(self.states[0])
        len_checked = len(self.checked)
        newly_added_states = set()

        new_len_checked = self.annotate_states(newly_added_states)
        while len_checked != new_len_checked:
            len_checked = new_len_checked
            new_len_checked = self.annotate_states(newly_added_states)

        for state in self.states:
            if '*' in state[0]:
                state_signature = state[0][1:]
            else:
                state_signature = state[0]
            if state_signature not in self.checked:
                state[0] = 'UNREACHABLE'

        self.states = list(filter(lambda x: x[0] != 'UNREACHABLE', self.states))

    def annotate_states(self, newly_added_states):
        for state_signature in self.checked:
            if state_signature == '':
                continue
            state = self.get_state_by_signature(state_signature)
            for signature in state:
                if '*' in signature:
                    newly_added_states.add(signature[-1])
                else:
                    newly_added_states.add(signature)
        self.checked = self.checked.union(newly_added_states)
        return len(self.checked)

    def get_state_by_signature(self, signature):
        for x in self.states:
            if x[0] in (signature, f'*{signature}'):
                return x

    def remove_deads(self):
        pass

    def create_state_error(self):
        """
            Maping blanks states and subscribe for state
        """
        self.create_state(state='<e>', state_final='*')
        for key_state, state in enumerate(self.states):
            for key_token, token in enumerate(state):
                if not token:
                    self.states[key_state][key_token] = '<e>'

    def compile(self):
        self.build_afnd()
        print_table(aut.states, [aut.sigma] + aut.alphabet_machine, 'autômato não determinizado')
        self.build_afd()
        print_table(aut.states, [aut.sigma] + aut.alphabet_machine, 'autômato determinizado')
        self.minimize()
        print_table(aut.states, [aut.sigma] + aut.alphabet_machine, 'autômato minimizado')
        self.create_state_error()
        print_table(aut.states, [aut.sigma] + aut.alphabet_machine, 'autômato com estado de erro')


words, grammar = formalize_data('tokens_grammar.txt')
# words = ['se', 'entao', 'senao', 'a', 'e', 'i', 'o', 'u']

aut = Automata(words=words, grammar=grammar)
aut.compile()
