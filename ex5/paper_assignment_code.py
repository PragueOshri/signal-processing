import numpy as np


# Code for section C
'''
This function finding the optimal bin size using the formula Hb = Ht - mean(Hnj)
It checking the sizes between 5 - 100, and return the bin that received the highest information
Input: Matrix of neurons recording, while each row represent different neuron 
Output: Bin Size
'''


def bin_for_burst_relate_info(train_mat):
    sizes_options = np.arange(5, 100, 1)
    list_Hb = []
    num_trials, len_trial = np.shape(train_mat)

    for s in sizes_options:
        probs_list = np.zeros(s + 1)
        # convert the mat to binned mat using different size bin each time
        # fill in a list of all the probabilities for each number of spikes in bin (between 0 to size bin)
        binned_mat = []
        for trial in range(num_trials):
            binned_trial = []
            for cell in range(len_trial):
                if cell >= len_trial - s:
                    #                     print(train_mat[trial][cell : cell + s])
                    num_spikes_in_bin = np.sum(train_mat[trial][cell: -1])
                else:
                    #                     print('in =', len(train_mat[trial][cell : cell + s]))
                    num_spikes_in_bin = np.sum(train_mat[trial][cell: (cell + s)])
                binned_trial.append(num_spikes_in_bin)

                # count the probabilities
                #                 print(s,num_spikes_in_bin)
                probs_list[num_spikes_in_bin] += 1
            binned_mat.append(binned_trial)

        total_num_cells = len(binned_mat[0]) * num_trials
        probs_list = np.array(probs_list) / total_num_cells

        # calculate Ht
        Ht = 0
        for p in probs_list:
            if p != 0:
                Ht += p * np.log2(p)
        Ht = -Ht

        # calculate mean Hnj
        noise_list_entropies = []
        first_non_zeros_probs = np.zeros(s + 1)
        saw_first_non_zero = False
        k = 1
        for b in range(len(binned_mat[0])):
            temp_probs_noise_list = np.zeros(s + 1)
            for t in range(num_trials):
                num_spikes = binned_mat[t][b]
                temp_probs_noise_list[num_spikes] += 1

            if temp_probs_noise_list[0] == (num_trials):
                k += 1
                first_non_zeros_probs[0] += num_trials
            else:
                if not saw_first_non_zero:
                    saw_first_non_zero = True
                    for i, p in enumerate(temp_probs_noise_list):
                        first_non_zeros_probs[i] += p
                else:
                    temp_probs_noise_list = np.array(temp_probs_noise_list) / (num_trials)
                    temp_Hn = 0
                    for p in temp_probs_noise_list:
                        if p != 0:
                            temp_Hn += p * np.log2(p)
                    temp_Hn = -temp_Hn
                    noise_list_entropies.append(temp_Hn)

        # calculate the Hn for the first non zero
        first_non_zeros_probs = np.array(first_non_zeros_probs) / (num_trials * k)
        temp_Hn = 0
        for p in first_non_zeros_probs:
            if p != 0:
                temp_Hn += p * np.log2(p)
        temp_Hn = -temp_Hn
        noise_list_entropies.append(temp_Hn)

        temp_mean_Hnj = np.mean(noise_list_entropies)

        # to allow comparison Ht and the averaging we divide the Ht by Hnj
        temp_correct_Ht = Ht / temp_mean_Hnj

        # calculate Hb
        temp_Hb = temp_correct_Ht - temp_mean_Hnj
        list_Hb.append(temp_Hb / s)  # normalize to bin size

    # find the maximal information
    maximal_Hb = np.max(list_Hb)
    maximal_Hb_index = np.where(list_Hb == maximal_Hb)

    # find the index to get the best size
    best_bin_size = sizes_options[maximal_Hb_index]

    return best_bin_size


# Code for section D
'''
This function generate simulated data using 3 Poisson neurons
Input:  1. Duration time in seconds (like 10)
        2. rate1 - baseline rate (number between 0 to 1 like 0.03)
        3. rate2 - activity rate (number between 0 to 1 like 0.5)
        4. switch time - between the two rates in sec (like 2)
Output: Matrix of 3 neurons, each row represent a neuron, column represented each sample (sample 1000Hz)
'''


def generate_data(duration, rate1, rate2, switch_time):
    samp = 1000
    rate1 = rate1
    rate2 = rate2
    duration = duration
    switch_time = switch_time

    neurons = []
    num_neurons = 3

    for n in range(num_neurons):
        temp_n = []
        i = 0
        iter_num = 1
        while i < duration * samp:

            # if I get to the switch time I replace between the rates
            if i == switch_time * iter_num * samp:
                iter_num += 1
                t = 0
                while t < switch_time * samp and i < duration * samp:
                    temp_spk = 1 if np.random.random() < rate2 else 0
                    temp_n.append(temp_spk)
                    t += 1
                    i += 1
            else:
                temp_spk = 1 if np.random.random() < rate1 else 0
                temp_n.append(temp_spk)
                i += 1
        neurons.append((temp_n))

    return np.array(neurons)

if __name__ == "__main__":
    my_generated_data = generate_data(10, 0.1, 0.7, 3)
    opt_bin_on_my_data = bin_for_burst_relate_info(my_generated_data)
    print(opt_bin_on_my_data)