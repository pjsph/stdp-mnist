import numpy as np
import random

code = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1]
code_probability = 0.5 #probability that the code is chosen as an input

def generate_random_code(length):
    """
    Generate a random sequence of 0s and 1s

    Parameters
    ----------
    length
        Length of the signal

    Returns
    -------
    rand_sig
        A random list consisting of 0s and 1s
    """
    rand_code = [np.random.randint(0, 2) for i in range(0,length-2)]
    rand_code.insert(0, 1)
    rand_code.append(1)
    return rand_code

def get_random_signal():
    rand_float = random.random()
    if rand_float < code_probability: #the code is chosen
        return ("c", signal_to_input(code, 0.05))
    else:  #a random signal is chosen
        rand_code = generate_random_code(len(code))
        return ("!c", signal_to_input(rand_code, 0.05))


def signal_to_input(sig, noise_level, h = 1):
    signal = [h * digit + np.random.normal(0, noise_level) for digit in sig]
    signal = [round(min(max(digit, 0), 1) * 255) for digit in signal]
    return signal


def get_labels():
    return ["c", "!c"]

