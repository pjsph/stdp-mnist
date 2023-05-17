import numpy as np

c1 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1]
c2 = [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]

draw = [('c1', c1), ('c2', c2)]
board = []

def get_random_signal():
    rand = np.random.randint(0, len(draw))
    for i in range(len(draw)):
        sig = signal_to_input(draw[rand][1], 0.05)
        for i in range(len(sig)):
                board.append(sig[i])
    return board
 
def signal_to_input(sig, noise_level, h = 1):
    signal = [h * digit + np.random.normal(0, noise_level) for digit in sig]
    signal = [round(min(max(digit, 0), 1) * 255) for digit in signal]
    return signal

def get_labels():
    return [d[0] for d in draw]

#=================This is just a sliding window where we try to check which code was uploaded=====================
#=============Does not work yet=================
# def sliding_window(window_size, is_code):
#    board = get_random_signal()
#    new_board = []
#    if len(board)<= window_size:
#     return board
#    else:
#        for i in range (len(board)-window_size+1):
#         new_board = (board[i:i+window_size])
#         for j in range (len(draw)):
#             n = 0
#             if board[j]==draw[1][j]:
#                 n+=1
#             if n > round(len(window_size)/2):
#                 is_code = True
#                 print ("The code is ", draw[0])
#    return (is_code,new_board)

# print(sliding_window(len(c1)))