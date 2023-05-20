import numpy as np
import random

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


#----------------------------- PARAMETERS

T = 16 #length of the signal (squared numbers to allow for plotting)

code = generate_random_code(T) # can be changed to a preferred, pre-defined code
code_probability = 0.5 #probability that the code is chosen as an input
print("Code utilisé : ", code)

#-----------------------------


def get_random_signal():
    """
    Randomly choose between the code or a random signal with noise

    Returns
    -------
    rand_sig
        A tuple consisting of the label "c" or "!c" at index 0
        and the output corresponding signal with noise at index 1
    """
    rand_float = random.random()
    if rand_float < code_probability: #the code is chosen
        return ("c", signal_to_input(code, 0.05))
    else:  #a random signal is chosen
        rand_code = generate_random_code(len(code))
        return ("!c", signal_to_input(rand_code, 0.05))


def signal_to_input(sig, noise_level, h = 1):
    """
    Add gaussian noise to the signal

    Parameters
    ----------
    sig
        The signal to be noised
    noise_level
    	Standard deviation of the noise
    h
    	Apply a multiplicator before noising (unused)
    
    Returns
    -------
    signal
        The noised signal
    """
    signal = [h * digit + np.random.normal(0, noise_level) for digit in sig]
    signal = [round(min(max(digit, 0), 1) * 255) for digit in signal]
    return signal


def get_labels():
    """
    Get the 2 labels (continuity with other version of main.py)
    
    Returns
    -------
    A list consisting of the 2 labels ("c" and "!c")
    """
    return ["c", "!c"]
    



