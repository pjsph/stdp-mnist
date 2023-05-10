import time
import mnist
import brian2 as b2
import brian2tools as b2tools
import numpy as np

def visualize_connectivity(S):
    """Plot synapses stored in S

    Parameters
    ----------
    S : b2.Synapses
        Synapses object to plot
    """
    Ns = len(S.source)
    Nt = len(S.target)
    b2.figure(figsize=(10, 4))
    b2.subplot(121)
    b2.plot(b2.zeros(Ns), b2.arange(Ns), 'ok', ms=10)
    b2.plot(b2.ones(Nt), b2.arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        b2.plot([0, 1], [i, j], '-k')
    b2.xticks([0, 1], ['Source', 'Target'])
    b2.ylabel('Neuron index')
    b2.xlim(-0.1, 1.1)
    b2.ylim(-1, max(Ns, Nt))
    b2.subplot(122)
    b2.plot(S.i, S.j, 'ok')
    b2.xlim(-1, Ns)
    b2.ylim(-1, Nt)
    b2.xlabel('Source neuron index')
    b2.ylabel('Target neuron index')
    b2.show()


def get_matrix_from_file(filename):
    """Load a weight matrix from hard disk

    Parameters
    ----------
    filename : str
        The file path

    Returns
    -------
    value_arr
        A 2D array representing the weight matrix
    """
    offset = 4 # indicates the number of characters to skip from the end of the filename used in determining the source and target dimensions of the weight matrix.
    if filename[-4-offset] == 'X':
        n_src = n_input
    else:
        if filename[-3-offset] == 'e':
            n_src = n_e
        else:
            n_src = n_i
    #This part is used to know the number of source neurons. If the character [-4-offset] is 'X'=> the weight matrix corresponds to the connections from input neurons. 

    if filename[-1-offset] == 'e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    #This part is used to know the number of target neurons. If the character [-1-offset] is 'e'=> the weight matrix corresponds to the connections which target the exhibitory neurons. 

    readout = np.load(filename)
    print(readout.shape, filename)
    # print(np.amax(readout[:,2]), np.amin(readout[:,2]))
    # This part loads the data from the filename using the "np.load" function
    # data is stored in "readout"
    
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    
    return value_arr
    # "value_arr" is a 2D array which is initialized with 0; the initial weight matrix
    # if the shape from "readout" data is not (0,), the values from "readout" will be assigned to the value_arr specific positions
    
def save_connections(ending = ''):
    """Save weight matrices

    Parameters
    ----------
    ending : str
        The suffix to add at the end of the file name
    """
    print('Saving connections...')
    conn_matrix = np.copy(synapses_input.w).reshape((n_input, n_e)) # Copy the weight matrix "synapses_input.w" using the "np.copy()" function. The weight matrix => 2D array with (n_input, n_e) dimensions; n_input - number of input neurons; n_e - number of excitatory neurons.
    matrix_input = [(i, j, conn_matrix[i,j]) for i in range(conn_matrix.shape[0]) for j in range(conn_matrix.shape[1])] # "matrix_input" is a list which stores the matrix weight connections as tuples

    np.save('weights/XeAe' + ending, matrix_input) # the list from above is saved in a file using "np.save" function with the extension ".npy"
    

def save_theta(ending = ''):
    """Save theta value

    Parameters
    ----------
    ending : str
        The suffix to add at the end of the file name
    """
    print('Saving theta...')
    np.save('weights/theta' + ending, neuron_group_e.theta)


def random_connections(path = 'random2/'):
    """Generate random weight matrices with the right size specified at the
    start of the simulation

    Parameters
    ----------
    path : str
        The folder in which to save the matrices
    """
    print('Randomizing matrices...')
    
    # matrix_ei -> connections from exhibitory neurons to inhibitory ones 
    matrix_ei = np.zeros((n_e, 3))
    for i in range(n_e):
        matrix_ei[i,:] = [i, i, 10.4]
    # Each row represents a connection from an excitatory neuron (i) to an inhibitory neuron (i), weight = 10.4

    # matrix_ie -> connections from inhibitory neurons to exhibitory ones
    matrix_ie = np.zeros((np.power(n_e, 2), 3)) # n_e - number of excitatiory neurons
    # the matrix is initialized with "n_e^2" and 3
    for k in range(np.power(n_e, 2)):
        i = k // n_e
        j = k % n_e
        matrix_ie[k] = [i, j, 0 if i == j else 17]
    # Each row represents a connection from an inhibitory neuron (i) to an excitatory neuron (j), weight=0 if i==j -> these are intrainhibitory connections else weight=17
     
     
    matrix_input = np.zeros((n_input * n_e, 3)) # n_e - number of excitatiory neurons
    # the matrix is initialized with "n_input * n_e" and 3
    for k in range(n_input * n_e):
        i = k % n_input
        j = k // n_input
        matrix_input[k] = [i, j, np.random.random() * 0.3 + 0.003]
    # Each row represents a connection from an input neuron (i) to an excitatory neuron (j), the weight is generated randomly 

    np.save(path + 'AeAi', matrix_ei)
    np.save(path + 'AiAe', matrix_ie)
    np.save(path + 'XeAe', matrix_input)
    # Each matrix is saved with different intuitive names

def normalize_weights():
    """Normalize weights so that the sum of all weights going from input neurons
    to a specific excitatory neuron is 78
    """
    conn_matrix = synapses_input.w # conn_matrix is the weight matrix between input neurons and excitatory neurons 
    temp_conn = np.copy(conn_matrix).reshape((n_input, n_e)) # "np.copy" copies the conn_matrix and is transformed into a 2D matrix with "reshape"
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = 78./colSums # we divide each column sum with 78 (the normalization value) and we obtain a factor => each factor is stored in "colFactors"
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j] # we approximate each weight column to 78
    synapses_input.w = temp_conn.flatten() # updates the weights of the connections between input neurons and excitatory neurons with the normalized values


def get_2d_input_weights():
    """Get input to excitatory neurons weight matrix in a format suitable
    for plotting
    """
    weight_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    conn_matrix = synapses_input.w
    weight_matrix = np.copy(conn_matrix).reshape((n_input, n_e))

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
            rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))

    return rearranged_weights


def plot_2d_input_weights():
    """Plot input to excitatory neurons weight matrix
        
    Returns
    -------
    im2
        Matplotlib image
    fig
        Matplotlib figure
    """
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = b2.figure(2, figsize = (18, 18))
    im2 = b2.imshow(weights, interpolation = 'nearest', vmin = 0, vmax = wmax, cmap = 'hot_r')
    b2.colorbar(im2)
    b2.title('weights of connection ' + name)
    fig.canvas.draw()
    #b2.pause(1)
    return im2, fig


def update_2d_input_weights(im, fig):
    """Update the input to excitatory neurons weight matrix plot
    Parameters
        im  
            Matplotlib image object 
        fig 
            Matplotlib figure object

    Returns
    -------
    im 
        Matplotlib image - It returns the updated image (im) as the result of the function
    """
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im


def get_current_performance(performance, current_example_num):
    """Get the network performance over the last 'update_interval' images

    Parameters
    ----------
    performance
        Array storing the performances
    current_example_num
        Index of the current image the network is training on

    Returns
    -------
    performance
        Array storing the performances, now updated
    """
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = output_numbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance


def plot_performance():
    """Plot the network performance

    Returns
    -------
    im2
        Matplotlib image
    performance
        Array storing 0s that will record future performances
    fig
        Matplotlib figure
    """
    num_evaluations = int(nb_examples/update_interval) + 1
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = b2.figure(3, figsize = (5, 5))
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance)
    b2.ylim(ymax = 100)
    b2.title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig


def update_performance_plot(im, performance, current_example_num, fig):
    """Update the network performance plot

    Parameters
    ----------
    im
        Matplotlib image
    performance
        Array storing the performances
    current_example_num
        Index of the current image the network is training on
    fig
        Matplotlib figure

    Returns
    -------
    im
        Matplotlib image
    performance
        Array storing the network performances, updated
    """
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance


def get_recognized_number_ranking(assignments, spike_rates):
    """Get the numbers recognized by the network for a given spike rate

    Parameters
    ----------
    assignments
        Array storing neuron assignments
    spike_rates
        Array storing the spike rate to analyse

    Returns
    -------
    array
        Array storing the recognized numbers sorted by their probability of
        being the actual number
    """
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]


def get_new_assignments(result_monitor, input_numbers):
    """Get neuron assignments (i.e class) based on given spike rates
    over given input numbers

    Parameters
    ----------
    result_monitor
        2D array storing, for each input number, the correspong spike rate
    input_numbers
        Array storing the input numbers the network has been fed with

    Returns
    -------
    assignments
        Array storing, for each excitatory neuron, the number it is more likely
        to spike to
    """
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in range(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments


# -----------------------------
# Load MNIST
# -----------------------------

print('Loading MNIST data...')

start = time.time()
training = mnist.get_labeled_data([0, 1], True)
end = time.time()
print('Loaded training set in:', end - start, "s")

start = time.time()
testing = mnist.get_labeled_data([0, 1], False)
end = time.time()
print('Loaded testing set in:', end - start, "s")

# ----------------------------
# Parameters & 2nd Layer Equations
# ----------------------------

test_mode = False

if test_mode:
    weight_path = 'weights/'
    nb_examples = 100 # 10000
    use_testing_set = True
    ee_STDP_on = False
    update_interval = nb_examples
else:
    weight_path = 'random2/'
    nb_examples = 60000
    use_testing_set = False
    ee_STDP_on = True

n_input = 784
n_e = 100 # 400
n_i = n_e
single_example_time = 0.35 * b2.second
resting_time = 0.15 * b2.second
runtime = nb_examples * (single_example_time + resting_time)
if nb_examples <= 10000:
    update_interval = 10 # nb_examples
    weight_update_interval = 20
else:
    update_interval = 10000
    weight_update_interval = 100
save_connections_interval = 1000

random_connections() # to be sure matrices have the right size

v_rest_e = -65. * b2.mV
v_rest_i = -60. * b2.mV
v_reset_e = -65. * b2.mV
v_reset_i = -45. * b2.mV
v_thresh_e = -52. * b2.mV
v_thresh_i = -40. * b2.mV
refrac_e = 5. * b2.ms
refrac_i = 2. * b2.ms

input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20 * b2.ms
tc_post1_ee = 20 * b2.ms
tc_post2_ee = 40 * b2.ms
nu_ee_pre = 0.0001
nu_ee_post = 0.01
wmax = 1.0

if test_mode:
    scr_e = 'v = v_reset_e; timer = 0 * second'
else:
    tc_theta = 1e7 * b2.ms
    theta_plus_e = 0.05 * b2.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0 * second'
offset = 20.0 * b2.mV

neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / (100 * ms) : volt
        I_synE = ge * nS * -v : amp
        I_synI = gi * nS * (-100. * mV - v) : amp
        dge/dt = -ge/(1.0 * ms) : 1
        dgi/dt = -gi/(2.0 * ms) : 1
        '''

if test_mode:
    neuron_eqs_e += '\n theta : volt'
else:
    neuron_eqs_e += '\n dtheta/dt = -theta/(tc_theta) : volt'
neuron_eqs_e += '\n dtimer/dt = 100.0e-3 : second'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE + I_synI) / nS) / (10 * ms) : volt
        I_synE = ge * nS * -v : amp
        I_synI = gi * nS * (-85. * mV - v) : amp
        dge/dt = -ge/(1.0 * ms) : 1
        dgi/dt = -gi/(2.0 * ms) : 1
        '''

eqs_stdp_ee = '''
        w : 1
        post2before : 1
        dpre/dt = -pre/(tc_pre_ee) : 1 (event-driven)
        dpost1/dt = -post1/(tc_post1_ee) : 1 (event-driven)
        dpost2/dt = -post2/(tc_post2_ee) : 1 (event-driven)
        '''
eqs_stdp_pre_ee = '''
        ge += w 
        pre = 1. 
        w = clip(w - nu_ee_pre * post1, 0, wmax)
        '''
eqs_stdp_post_ee = '''
        post2before = post2 
        w = clip(w + nu_ee_post * pre * post2before, 0, wmax) 
        post1 = 1. 
        post2 = 1. 
        '''

#b2.ion()
result_monitor = np.zeros((update_interval, n_e))

print('Creating neuron groups...')

neuron_group_e = b2.NeuronGroup(n_e, neuron_eqs_e, threshold='(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)', reset=scr_e, refractory=refrac_e)
neuron_group_i = b2.NeuronGroup(n_i, neuron_eqs_i, threshold='v > v_thresh_i', reset='v = v_reset_i', refractory=refrac_i)

neuron_group_e.v = v_rest_e - 40. * b2.mV
neuron_group_i.v = v_rest_i - 40. * b2.mV
if test_mode:
    neuron_group_e.theta = np.load(weight_path + 'theta.npy') * b2.volt
else:
    neuron_group_e.theta = np.ones((n_e)) * 20.0 * b2.mV

print('Creating synapses...')

weight_matrix_ei = get_matrix_from_file('random2/AeAi.npy')
synapses_ei = b2.Synapses(neuron_group_e, neuron_group_i, 'w : 1', on_pre='ge += w')
synapses_ei.connect()
synapses_ei.w = weight_matrix_ei.flatten()

weight_matrix_ie = get_matrix_from_file('random2/AiAe.npy')
synapses_ie = b2.Synapses(neuron_group_i, neuron_group_e, 'w : 1', on_pre='gi += w')
synapses_ie.connect()
synapses_ie.w = weight_matrix_ie.flatten()

# visualize_connectivity(synapses_ie) # use for visualisation

rate_monitor_e = b2.PopulationRateMonitor(neuron_group_e) 
rate_monitor_i = b2.PopulationRateMonitor(neuron_group_i)
spike_counter = b2.SpikeMonitor(neuron_group_e, record=False)

spike_monitor_e = b2.SpikeMonitor(neuron_group_e)
spike_monitor_i = b2.SpikeMonitor(neuron_group_i)

#b2.figure(1)
##b2.ion()
#b2.subplot(211)
#b2tools.brian_plot(spike_monitor_e)
#b2.subplot(212)
#b2tools.brian_plot(spike_monitor_i)
#b2.draw()
#b2.pause(0.01)

# ----------------------------
# Input Layer Equations
# ----------------------------

print('Creating input layer neuron group and synapses...')

input_group = b2.PoissonGroup(n_input, 0 * b2.Hz)
rate_monitor_input = b2.PopulationRateMonitor(input_group)

weight_matrix_input = get_matrix_from_file(weight_path + 'XeAe.npy')
if ee_STDP_on:
    synapses_input = b2.Synapses(input_group, neuron_group_e, eqs_stdp_ee, on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee)
else:
    synapses_input = b2.Synapses(input_group, neuron_group_e, 'w : 1', on_pre='ge += w')
synapses_input.connect()
synapses_input.w = weight_matrix_input.flatten()
synapses_input.delay = 'rand()*10*ms'

# ----------------------------
# Simulation
# ----------------------------

previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * nb_examples
output_numbers = np.zeros((nb_examples, 10))

if not test_mode:
    input_weight_monitor, fig_weights = plot_2d_input_weights()

performance_monitor, performance, fig_performance = plot_performance()

input_group.rates = 0

b2.run(0 * b2.ms)

j = 0
while j < int(nb_examples):
    if test_mode:
        if use_testing_set:
            rates = [col / 8. * input_intensity * b2.Hz for row in testing[0][j%10000] for col in row]
        else:
            rates = [col / 8. * input_intensity * b2.Hz for row in training[0][j%60000] for col in row]
    else:
        normalize_weights()
        rates = [col / 8. * input_intensity * b2.Hz for row in training[0][j%60000] for col in row]

    input_group.rates = rates

    b2.run(single_example_time, report='text')

    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
    if j % weight_update_interval == 0 and not test_mode:
        update_2d_input_weights(input_weight_monitor, fig_weights)
    if j % save_connections_interval == 0 and j > 0 and not test_mode:
        save_connections(str(j))
        save_theta(str(j))

    current_spike_count = np.asarray(spike_counter.count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counter.count[:])
    print('spiked', sum(current_spike_count), 'times')
    if sum(current_spike_count) < 5:
        print('-- Increased intensity')
        input_intensity += 1
        input_group.rates = 0
        b2.run(resting_time)
    else:
        print('-- OK')
        result_monitor[j%update_interval,:] = current_spike_count
        if test_mode and use_testing_set:
            input_numbers[j] = testing[1][j%10000]
        else:
            input_numbers[j] = training[1][j%60000]

        output_numbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])

        print(j)
        if j % 10 == 0 and j > 0:
            print('Runs done:', j, 'of', nb_examples)
        if j % update_interval == 0 and j > 0:
            unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
            print('Classification performance', performance[:(j//update_interval) + 1])

        input_group.rates = 0
        b2.run(resting_time)
        input_intensity = start_input_intensity
        j += 1
