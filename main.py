import time

from brian2 import rand, synapses
import mnist
import brian2 as b2
import brian2tools as b2tools
import numpy as np

def visualize_connectivity(S):
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
    print('Saving connections...')
    conn_matrix = np.copy(synapses_input.w).reshape((n_input, n_e))
    matrix_input = [(i, j, conn_matrix[i,j]) for i in range(conn_matrix.shape[0]) for j in range(conn_matrix.shape[1])]

    np.save('weights/XeAe' + ending, matrix_input)
    

def save_theta(ending = ''):
    print('Saving theta...')
    np.save('weights/theta' + ending, neuron_group_e.theta)


def random_connections(path = 'random2/'):
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

def normalize_weights():
    conn_matrix = synapses_input.w
    temp_conn = np.copy(conn_matrix).reshape((n_input, n_e))
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = 78./colSums
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    synapses_input.w = temp_conn.flatten()


def get_2d_input_weights():
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weigths = np.zeros((num_values_col, num_values_row))
    conn_matrix = synapses_input.w
    weight_matrix = np.copy(conn_matrix).reshape((n_input, n_e))

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
            rearranged_weigths[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))

    return rearranged_weigths


def plot_2d_input_weights():
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
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im


def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = output_numbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance


def plot_performance():
    num_evaluations = int(nb_examples/update_interval)
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
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == 1)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
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


# ------------------------------
# Load MNIST
# -----------------------------

print('Loading MNIST data...')

start = time.time()
training = mnist.get_labeled_data([0, 1], True)
end = time.time()
print('Loaded training set in:', end - start, "s")

# ----------------------------
# Parameters & 2nd Layer Equations
# ----------------------------
rate_monitors={}
nb_examples = 200

n_input = 784
n_e = 100 # 400
n_i = n_e
single_example_time = 0.35 * b2.second
resting_time = 0.15 * b2.second
runtime = nb_examples * (single_example_time + resting_time)
update_interval = 100 # nb_examples
weight_update_interval = 50
save_connections_interval = 100

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
print(rate_monitor_e)
rate_monitor_i = b2.PopulationRateMonitor(neuron_group_i)
spike_counter = b2.SpikeMonitor(neuron_group_e, record=False)

# for i in range (nb_examples):
# rate_monitors = rate_monitor_e 
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

weight_matrix_input = get_matrix_from_file('random2/XeAe.npy')
synapses_input = b2.Synapses(input_group, neuron_group_e, eqs_stdp_ee, on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee)
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

input_weight_monitor, fig_weights = plot_2d_input_weights()

performance_monitor, performance, fig_performance = plot_performance()

input_group.rates = 0

b2.run(0 * b2.ms)

j = 0
while j < int(nb_examples):
    normalize_weights()

    rates = [col / 8. * input_intensity * b2.Hz for row in training[0][j%60000] for col in row]
    input_group.rates = rates

    b2.run(single_example_time, report='text')

    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
    if j % weight_update_interval == 0:
        update_2d_input_weights(input_weight_monitor, fig_weights)
    if j % save_connections_interval == 0 and j > 0:
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
        input_numbers[j] = training[1][j%60000]

        output_numbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])

        if j % 10 == 0 and j > 0:
            print('Runs done:', j, 'of', nb_examples)
        if j % update_interval == 0 and j > 0:
            unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
            print('Classification performance', performance[:(j//update_interval) + 1])

        input_group.rates = 0
        b2.run(resting_time)
        input_intensity = start_input_intensity
        j += 1

# -------------------
# Results saving
# -------------------
data_path = './'
print ('save results')
np.save(data_path + 'activity/resultPopVecs' + str(nb_examples), result_monitor)
np.save(data_path + 'activity/inputNumbers' + str(nb_examples), input_numbers)
# -------------------
# Results plotting
# -------------------
fig_num=0

if spike_counter:
    b2.figure(fig_num)
    fig_num += 1

    for i in enumerate(spike_counter):
        b2.subplot(len(spike_counter), 1, i)
        b2.plot(spike_counter['Ae'].count[:])
        b2.title('Spike count of population ')

plot_2d_input_weights()
b2.ioff()
b2.show()

if rate_monitor_e:
    b2.figure(fig_num)
    fig_num += 1

    for i in enumerate(rate_monitor_e):
        b2.subplot(len(rate_monitor_e), 1, i)
        b2.plot(rate_monitor_e.times/b2.second, rate_monitor_e.rate, '.')
        b2.title('Rates of population excitatory neurons')

if rate_monitor_i:
    b2.figure(fig_num)
    fig_num += 1

    for i in enumerate(rate_monitor_i):
        b2.subplot(len(rate_monitor_i), 1, i)
        b2.plot(rate_monitor_i.times/b2.second, rate_monitor_i.rate, '.')
        b2.title('Rates of population inhibitory neurons')
    
if spike_monitor_e:
    b2.figure(fig_num)
    fig_num += 1

    for i in enumerate(spike_monitor_e):
        b2.subplot(len(spike_monitor_e), 1, i)
        b2.raster_plot(spike_monitor_e)
        b2.title('Spikes of excitatory population')

if spike_monitor_i:
    b2.figure(fig_num)
    fig_num += 1

    for i in enumerate(spike_monitor_i):
        b2.subplot(len(spike_monitor_i), 1, i)
        b2.raster_plot(spike_monitor_i)
        b2.title('Spikes of inhibitory population')
    