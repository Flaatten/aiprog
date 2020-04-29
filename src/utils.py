import random


def generate_random_int_in_range(from_including, to_excluding):
    return random.randint(from_including, to_excluding - 1)

def increment(dictionary, key):
    dictionary[key] = dictionary[key] + 1