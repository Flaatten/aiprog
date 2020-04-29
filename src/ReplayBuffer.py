import random


class ReplayBuffer():

    def __init__(self):
        self.cases = []

    def add_case(self, case):
        self.cases.append(case)

    def get_mini_batch(self, n_elements):
        if len(self.cases) < n_elements:
            return self.cases
            #raise ValueError("Trying to request " + str(n_elements) + " elements from ReplayBuffer, which consists of " + str(len(self.cases)) + " cases.")
        return random.sample(self.cases, n_elements)

    def clear(self):
        self.cases.clear()