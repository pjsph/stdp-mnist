import time

from matplotlib import contextlib
import mnist
import signals
import brian2 as b2
import brian2tools as b2tools
import numpy as np
from multiprocessing import Process, Queue
from matplotlib.animation import FuncAnimation

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
    offset = 4
    if filename[-4-offset] == 'X':
        n_src = n_input
    else:
        if filename[-3-offset] == 'e':
            n_src = n_e
        else:
            n_src = n_i

    if filename[-1-offset] == 'e':
        n_tgt = n_e
    else:
        n_tgt = n_i

    readout = np.load(filename)
    print(readout.shape, filename)
    # print(np.amax(readout[:,2]), np.amin(readout[:,2]))

    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    
    return value_arr


def save_connections(ending = ''):
    """Save weight matrices

    Parameters
    ----------
    ending : str
        The suffix to add at the end of the file name
    """
    print('Saving connections...')
    conn_matrix = np.copy(synapses_input.w).reshape((n_input, n_e))
    matrix_input = [(i, j, conn_matrix[i,j]) for i in range(conn_matrix.shape[0]) for j in range(conn_matrix.shape[1])]

    np.save('weights/XeAe' + ending, matrix_input)
    

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
    matrix_ei = np.zeros((n_e, 3))
    for i in range(n_e):
        matrix_ei[i,:] = [i, i, 10.4]

    matrix_ie = np.zeros((np.power(n_e, 2), 3))
    for k in range(np.power(n_e, 2)):
        i = k // n_e
        j = k % n_e
        matrix_ie[k] = [i, j, 0 if i == j else 17]

    matrix_input = np.zeros((n_input * n_e, 3))
    for k in range(n_input * n_e):
        i = k % n_input
        j = k // n_input
        matrix_input[k] = [i, j, np.random.random() * 0.3 + 0.003]

    np.save(path + 'AeAi', matrix_ei)
    np.save(path + 'AiAe', matrix_ie)
    np.save(path + 'XeAe', matrix_input)

def normalize_weights(sum_value=78.0):
    """Normalize weights so that the sum of all weights going from input neurons
    to a specific excitatory neuron is the number of input neurons
    divided by the number of classes
    """
    conn_matrix = synapses_input.w
    temp_conn = np.copy(conn_matrix).reshape((n_input, n_e))
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = float(sum_value)/colSums
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    synapses_input.w = temp_conn.flatten()


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


def animate_2d_input_weights(q, n_input, n_e, wmax):
    name = 'XeAe'
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    weights = np.zeros((n_e_sqrt*n_in_sqrt, n_e_sqrt*n_in_sqrt))
    fig = b2.figure(2, figsize = (18, 18))
    im = b2.imshow(weights, interpolation = 'nearest', vmin = 0, vmax = wmax, cmap = 'hot_r')
    b2.colorbar(im)
    b2.title('weights of connection ' + name)

    def _animate(frame):
        while not q.empty():
            weights = q.get()
            im.set_array(weights)

    animation = FuncAnimation(fig, _animate, interval=500)
    b2.show()


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
    correct = len(np.where(output_numbers[start_num:end_num, 0] == input_numbers[start_num:end_num])[0])
    print('=============', correct)
    print(output_numbers, "///////", input_numbers)
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance


def animate_performance_plot(q, time_steps, performance):
    fig = b2.figure(3, figsize = (5, 5))
    ax = fig.add_subplot(111)
    im, = ax.plot(time_steps, performance)
    b2.ylim(ymax = 100)
    b2.title('Classification performance')

    def _animate(frame):
        while not q.empty():
            performance = q.get()
            im.set_ydata(performance)

    animation = FuncAnimation(fig, _animate, interval=500, cache_frame_data=False)
    b2.show()


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
    labels = signals.get_labels()
    summed_rates = [0] * len(labels)
    for i in range(len(labels)):
        num_assignments = len(np.where(assignments == labels[i])[0])
        if num_assignments > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == labels[i]]) / num_assignments
    return np.asarray(labels)[np.argsort(summed_rates)[::-1]]


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
    assignments = np.empty(n_e, str)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in signals.get_labels():
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
            for i in range(n_e):
                if rate[i] > maximum_rate[i]:
                    maximum_rate[i] = rate[i]
                    assignments[i] = j
    return assignments

def animate_redo_plot(q, runs, redos, n_e, n_input, base_intensity):
    fig = b2.figure(3, figsize = (5, 5))
    ax = fig.add_subplot(111)
    im, = ax.plot(runs, redos)
    b2.ylim(ymax = 100)
    b2.title("Amount of redos each "+str(len(runs))+" runs \nfor "+str(n_input)+" input neurons, "+str(n_e)+" exc neurons, \nand a base intensity of "+str(base_intensity))

    def _animate(frame):
        while not q.empty():
            redos = q.get()
            im.set_ydata(redos)

    animation = FuncAnimation(fig, _animate, interval=500, cache_frame_data=False)
    print("Pourtant ça marche...")
    b2.show()
  
if __name__ == "__main__":

    # ----------------------------
    # Parameters & 2nd Layer Equations
    # ----------------------------

    #only the maximum number of images loaded in the memory.
    #if not all the 0-9 digits are used, the ACtual number of
    #loaded images will be lesser than this maximum
    loaded_training_images = 60000 #60000 (ou None)
    loaded_testing_images = 10000 #10000 (ou None)
    
    test_mode = False

    if test_mode:
        weight_path = 'weights/'
        nb_examples = 10001 # 10000
        use_testing_set = True
        ee_STDP_on = False
        update_interval = 1000 # nb_examples
    else:
        weight_path = 'random2/'
        nb_examples = 10001 # 60000
        use_testing_set = False
        ee_STDP_on = True

    if loaded_training_images and loaded_testing_images:
        if max(loaded_training_images,loaded_testing_images) < nb_examples:
            print("ERREUR:  Nombre d'examples présentés inférieur au nombre d'images chargées")
        
    n_input = 16
    n_e = 100 # 400
    n_i = n_e
    minimum_spike_number = 2
    single_example_time = 0.35 * b2.second
    resting_time = 0.15 * b2.second
    runtime = nb_examples * (single_example_time + resting_time)
    if nb_examples <= 10000:
        update_interval = 10 # nb_examples
        weight_update_interval = 10
    else:
        update_interval = 100
        weight_update_interval = 50
    save_connections_interval = 500

    random_connections() # to be sure matrices have the right size

    v_rest_e = -65. * b2.mV
    v_rest_i = -60. * b2.mV
    v_reset_e = -65. * b2.mV
    v_reset_i = -45. * b2.mV
    v_thresh_e = -52. * b2.mV
    v_thresh_i = -40. * b2.mV
    refrac_e = 5. * b2.ms
    refrac_i = 2. * b2.ms

    input_intensity = 2. #2
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

    # -----------------------------
    # Load signals
    # -----------------------------

    print('Loading signals data...')

#    start = time.time()
#    training = mnist.get_labeled_data([0, 1], True)
#    end = time.time()
#    print('Loaded training set in:', end - start, "s")
#
#    start = time.time()
#    testing = mnist.get_labeled_data([0, 1], False)
#    end = time.time()
#    print('Loaded testing set in:', end - start, "s")
    
    # ----------------------------
    # Simulation
    # ----------------------------

    previous_spike_count = np.zeros(n_e)
    assignments = np.empty(n_e, str)
    input_numbers = [0] * nb_examples
    output_numbers = np.empty((nb_examples, len(signals.get_labels())), np.dtype('U3'))

    if not test_mode:
        weight_data_queue = Queue()
        weight_plot_process = Process(target=animate_2d_input_weights, args=(weight_data_queue, n_input, n_e, wmax))
        weight_plot_process.start()

    #performance_monitor, performance, fig_performance = plot_performance()
    num_evaluations = int(nb_examples/update_interval) + 1
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    performance_data_queue = Queue()
    performance_plot_process = Process(target=animate_performance_plot, args=(performance_data_queue, time_steps, performance))
    performance_plot_process.start()

    #record how many times the minimum amount of spikes wasn't reached for each run
    redos = np.zeros(nb_examples)
    redo_display_interval = 100
    redos_every_interval = np.zeros(int(nb_examples/redo_display_interval))
    redo_data_queue = Queue()
    redo_interval_runs = range(0, int(nb_examples/redo_display_interval))
    redo_plot_process = Process(target=animate_redo_plot, args=(redo_data_queue, redo_interval_runs, redos_every_interval, n_e, n_input, input_intensity))
    redo_plot_process.start()
    current_is_redo = False

    input_group.rates = 0

    b2.run(0 * b2.ms)

    number_of_labels = len(signals.get_labels())

    j = 0
    signal = signals.get_random_signal()
    while j < int(nb_examples):

        if not current_is_redo: #if it's a redo the input must stay the same
            signal = signals.get_random_signal()
        
        rates = [digit / 8. * input_intensity * b2.Hz for digit in signal[1]]

        if not test_mode:
            normalize_weights(8.)

        input_group.rates = rates

        b2.run(single_example_time, report='text')

        if j % update_interval == 0 and j > 0:
            assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
        if j % weight_update_interval == 0 and not test_mode:
            weight_data_queue.put(get_2d_input_weights())
        if j % save_connections_interval == 0 and j > 0 and not test_mode:
            save_connections(str(j))
            save_theta(str(j))

        current_spike_count = np.asarray(spike_counter.count[:]) - previous_spike_count
        previous_spike_count = np.copy(spike_counter.count[:])
        print('spiked', sum(current_spike_count), 'times')
        if sum(current_spike_count) < minimum_spike_number:
            print('-- Increased intensity')
            input_intensity += 1
            input_group.rates = 0
            b2.run(resting_time)
            redos[j] += 1
        else:
            print('-- OK')
            result_monitor[j%update_interval,:] = current_spike_count
            input_numbers[j] = signal[0]

            output_numbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])

            print("Run",j,"done")
            if j % 10 == 0 and j > 0:
                print('Runs done:', j, 'of', nb_examples)
            if j % update_interval == 0 and j > 0:
                performance = get_current_performance(performance, j)
                performance_data_queue.put(performance)
                print('Classification performance', performance[:(j//update_interval) + 1])

            input_group.rates = 0
            b2.run(resting_time)
            input_intensity = start_input_intensity

            if ((j+1) % redo_display_interval == 0) and j > 0:
                redos_every_interval[int((j+1)/redo_display_interval)-1] = sum(redos[(j-redo_display_interval+1):(j+1)])
                redo_data_queue.put(redos_every_interval)

            j += 1
        

    if not test_mode:
        weight_plot_process.join()
    performance_plot_process.join()
    redo_plot_process.join()
