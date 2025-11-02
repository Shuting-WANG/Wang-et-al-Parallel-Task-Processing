import numpy as np
import matplotlib.pyplot as plt
import torch

def dataset(dt, batch_size=16, DT = False):
    """
    Generates a synthetic dataset for single motor task, single cognitive task, and dual-task.

    Parameters:
    dt (float): Time step size in milliseconds.
    batch_size (int): Number of trials to generate in the batch.
    DT (bool): If True, generates dual-task trials; if False, generates single-task trials.
    """

    T, T_cue, T_lick, T_lever, T_delay, T_bet = 8000, 1000, 200, 4500, 500, 1500
    all_inputs, all_targets, all_CD_masks, all_dual_motor_masks, all_trial_onset, all_trial_type = [], [], [], [], [], []

    motor_tuning, cog_tuning, rule_tuning = create_tuning_functions()
    
    for _ in range(batch_size):
        # randomly select a time point as start; leave 1000 ms at the beginning and end of the trial
        t_step_start = int(np.random.uniform(1000 / dt, (T - T_lever - T_delay - 1000) / dt)) 
        # initiate output arrays
        single_input = np.random.normal(0, np.sqrt(2/0.5)*0.1, size=(round(T / dt), 42)).astype(np.float32) # stimulus input + context
        single_target = np.zeros((round(T / dt), 2)).astype(np.float32) # output: lever and GNG
        CD_mask = np.zeros((round(T / dt), 2)).astype(np.float32) # phase mask for calculating coding direction: 1=baseline, 2=motor, 3=Go, 4=No-Go
        dual_motor_mask = np.zeros((round(T / dt), 2)).astype(np.float32) # dual-task period for evaluating motor performance
       
        # Randomly pick a task: 1: ST-Lever, 2: ST-Go, 3: ST-No-Go, 4: DT-Go, 5: DT-No-Go
        if DT:
            trial_type = np.random.choice([1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]) # 25% ST-Lever, 25% ST-GNG, 50% DT
        else:
            trial_type = np.random.choice([1, 2, 3])

        LEVER, GNG = None, None
        if trial_type == 1 or trial_type == 4 or trial_type == 5:
            LEVER = 1
            single_input += rule_tuning[:,0]    # rule 0: Context-Lever
        if trial_type == 2 or trial_type == 4:
            GNG = 1    # Go trial
            single_input += rule_tuning[:,1]    # rule 1: Context-GNG
        if trial_type == 3 or trial_type == 5:
            GNG = 2    # No-Go trial
            single_input += rule_tuning[:,1]    # rule 1: Context-GNG
        if trial_type == 4 or trial_type == 5:
            single_input += rule_tuning[:,2]    # rule 2: Context-DT
            dual_motor_mask[t_step_start + round((T_delay + T_bet) / dt) : t_step_start + round((T_delay + T_lever) / dt), 0] = 1  # from initiation of GNG task until completion of lever task
            if trial_type == 4:
                CD_mask[t_step_start + round((T_delay+T_bet) / dt):t_step_start + round((T_delay+T_bet+T_cue+T_lick) / dt),1] = 3  # Go cue presentation + response window
            if trial_type == 5:
                CD_mask[t_step_start - round(1000/dt) : t_step_start, 0] = 1                           # in DT-No-Go, baseline [-1,0]s
                CD_mask[t_step_start : t_step_start + round((T_delay + T_lever) / dt), 0] = 2          # in DT-No-Go, motor task window
                CD_mask[t_step_start + round((T_delay+T_bet) / dt):t_step_start + round((T_delay+T_bet+T_cue+T_lick) / dt),1] = 4  # No-Go cue presentation + response window

        if LEVER == 1:
            single_input[t_step_start : t_step_start + round(T_lever / dt), :] += motor_tuning[:,0]    # motor stimulus input
            single_target[t_step_start + round(T_delay / dt) : t_step_start + round((T_delay + T_lever) / dt), 0] = 1    # motor output: lever position
        
        if GNG == 1:
            single_input[t_step_start + round((T_delay+T_bet) / dt):t_step_start + round((T_delay+T_bet+T_cue) / dt), :] += cog_tuning[:,0]    # Go stimulus input
            single_target[t_step_start + round((T_delay+T_bet+T_cue) / dt):t_step_start + round((T_delay+T_bet+T_cue+T_lick) / dt),1] = 1      # GNG output: lick rate    

        if GNG == 2:
            single_input[t_step_start + round((T_delay+T_bet) / dt):t_step_start + round((T_delay+T_bet+T_cue) / dt), :] += cog_tuning[:,1]    # No-Go stimulus input

        # relu
        single_input = np.maximum(single_input, 0)
    
        all_inputs.append(single_input)
        all_targets.append(single_target)
        all_CD_masks.append(CD_mask)
        all_dual_motor_masks.append(dual_motor_mask)
        all_trial_onset.append(t_step_start)
        all_trial_type.append(trial_type)

    # Stack each trial along batch dimension -> (time, batch, features)
    all_inputs = np.stack(all_inputs, axis=1)
    all_targets = np.stack(all_targets, axis=1)
    all_CD_masks = np.stack(all_CD_masks, axis=1)
    all_dual_motor_masks = np.stack(all_dual_motor_masks, axis=1)

    return all_inputs, all_targets, all_CD_masks, all_dual_motor_masks, all_trial_onset, all_trial_type


def create_tuning_functions(num_motor_stimuli=1, num_motor_tuned=24,num_cog_stimuli=2, num_cog_tuned=12, num_rules=3, num_rule_tuned=6, kappa=2):
    """
    Creates tuning functions for motor, cognitive, and rule stimuli.

    Returns:
        cog_tuning (ndarray): Array of shape (n_input, num_cog_stimuli), representing cognitive stimulus tuning for each input neuron.
        motor_tuning (ndarray): Array of shape (n_input, num_motor_stimuli), representing motor stimulus tuning for each input neuron.
        rule_tuning (ndarray): Array of shape (n_input, num_rules), representing rule/context tuning for each input neuron.

    The returned arrays are used to modulate input neurons according to task context and stimulus type.
    """
    n_input = num_motor_tuned + num_cog_tuned + num_rule_tuned  # total number of input-tuned neurons
    motor_tuning = np.zeros((n_input, num_motor_stimuli))
    cog_tuning = np.zeros((n_input, num_cog_stimuli))
    rule_tuning = np.zeros((n_input, num_rules))

    motor_pref = np.float32(np.arange(0,360,360/num_motor_tuned)) 
    motor_stim = np.float32(np.arange(0,360,360/num_motor_stimuli))
    cog_pref = np.float32(np.arange(0,360,360/num_cog_tuned))
    cog_stim = np.float32(np.arange(0,360,360/num_cog_stimuli))

    for n in range(num_motor_tuned):
        d = np.cos((motor_stim[0] - motor_pref[n])/180*np.pi)
        motor_tuning[n,0] = 4*np.exp(kappa*d)/np.exp(kappa)

    for n in range(num_cog_tuned):
        for i in range(num_cog_stimuli):
            d = np.cos((cog_stim[i] - cog_pref[n])/180*np.pi)
            cog_tuning[num_motor_tuned+n,i] = 4*np.exp(kappa*d)/np.exp(kappa)

    for n in range(num_rule_tuned):
        for i in range(num_rules):
            if n%num_rules == i:
                rule_tuning[num_cog_tuned+num_motor_tuned+n,i] = 4

    return motor_tuning, cog_tuning, rule_tuning


def plot_behav_per(win,dt,num_trial,trial_infos,performance_dict,task_onsets):
    """
    Visualizes behavioral performance for different trial types and returns aligned performance data.

    Parameters:
        win (list or array): Time window (in seconds) to align and plot performance.
        dt (float): Time step size in milliseconds.
        num_trial (int): Number of trials.
        trial_infos (list or array): List of trial type identifiers for each trial.
        performance_dict (dict): Dictionary mapping trial indices to performance arrays.
        task_onsets (list or array): List of onset times (indices) for each trial.
    Returns:
        aligned_performance (list): List of aligned performance arrays for each trial.
    """

    win = np.array(win)*1000/dt
    win = win.astype(int)

    aligned_performance = [performance_dict[i][task_onsets[i][0]+win[0]:task_onsets[i][0]+win[1],:] for i in range(num_trial)]

    stLEVER_performance = np.array([aligned_performance[i] for i in range(num_trial) if trial_infos[i][0] == 1])
    stGO_performance = np.array([aligned_performance[i] for i in range(num_trial) if trial_infos[i][0] == 2])
    stNG_performance = np.array([aligned_performance[i] for i in range(num_trial) if trial_infos[i][0] == 3])
    dtGO_performance = np.array([aligned_performance[i] for i in range(num_trial) if trial_infos[i][0] == 4])
    dtNG_performance = np.array([aligned_performance[i] for i in range(num_trial) if trial_infos[i][0] == 5])

    t = np.arange(win[0], win[1]) * dt / 1000
    _, ax = plt.subplots(2,5,figsize=(12, 5), sharex=True, sharey=True)
    ax[0,0].plot(t, stLEVER_performance[:,:,0].T, 'r', alpha=0.1)
    ax[0,1].plot(t, stGO_performance[:,:,0].T, 'r', alpha=0.1)
    ax[0,2].plot(t, stNG_performance[:,:,0].T, 'r', alpha=0.1)
    ax[0,3].plot(t, dtGO_performance[:,:,0].T, 'r', alpha=0.1)
    ax[0,4].plot(t, dtNG_performance[:,:,0].T, 'r', alpha=0.1)

    ax[1,0].plot(t, stLEVER_performance[:,:,1].T, 'g', alpha=0.1)
    ax[1,1].plot(t, stGO_performance[:,:,1].T, 'g', alpha=0.1)
    ax[1,2].plot(t, stNG_performance[:,:,1].T, 'g', alpha=0.1)
    ax[1,3].plot(t, dtGO_performance[:,:,1].T, 'g', alpha=0.1)
    ax[1,4].plot(t, dtNG_performance[:,:,1].T, 'g', alpha=0.1)

    for l in [0,3,4]:
        ax[0,l].vlines(0, -0.2, 1.2, 'gray', '--')
        ax[1,l].vlines(0, -0.2, 1.2, 'gray', '--')
        ax[0,l].vlines(5, -0.2, 1.2, 'gray', '--')
        ax[1,l].vlines(5, -0.2, 1.2, 'gray', '--')

    for g in [1,2,3,4]:
        ax[0,g].vlines(2, -0.2, 1.2, 'gray', '--')
        ax[1,g].vlines(2, -0.2, 1.2, 'gray', '--')
        ax[0,g].vlines(3, -0.2, 1.2, 'gray', '--')
        ax[1,g].vlines(3, -0.2, 1.2, 'gray', '--')

    ax[0,0].set_title('ST-Lever')
    ax[0,1].set_title('ST-Go')
    ax[0,2].set_title('ST-No-Go')
    ax[0,3].set_title('DT-Go')
    ax[0,4].set_title('DT-No-Go')

    ax[0,0].set_ylabel('Lever predictions')
    ax[1,0].set_ylabel('GNG predictions')

    plt.tight_layout()
    plt.show()

    return aligned_performance


def sort_activity(activity,sort_idx=None):
    if sort_idx is None:
        peak_time = np.argmax(activity, axis=0)
        sort_idx = np.argsort(peak_time)
    activity = activity[:,sort_idx]
    return activity, sort_idx

def plot_activity(win,dt,num_trial,trial_infos,activity_dict,task_onsets, sort_task=None, normalize=True, neuron_type = 'all'):
    """
    Plots trial-averaged neural activity for different trial types, with options for sorting and normalization.

    Parameters:
        win (list or array): Time window (in seconds) to align and plot activity.
        dt (float): Time step size in milliseconds.
        num_trial (int): Number of trials.
        trial_infos (list or array): List of trial type identifiers for each trial.
        activity_dict (dict): Dictionary mapping trial indices to neural activity arrays.
        task_onsets (list or array): List of onset times (indices) for each trial.
        sort_task (str or None): Task condition to use for sorting neurons ('self', 'stLEVER', 'stGO', 'stNG', 'dtGO', 'dtNG'), or None for no sorting.
        normalize (bool): Whether to normalize activity by the maximum value across all conditions.
        neuron_type (str): Type of neurons to plot ('all', 'exc', 'inh').
    Returns:
        aligned_activity (list): List of aligned activity arrays for each trial.
    """

    win = np.array(win)*1000/dt
    win = win.astype(int)
    t = np.arange(win[0], win[1]) * dt / 1000
    n_neurons = activity_dict[0].shape[1]

    aligned_activity = [activity_dict[i][task_onsets[i][0]+win[0]:task_onsets[i][0]+win[1],:] for i in range(num_trial)]
    stLEVER_activity = np.array([aligned_activity[i] for i in range(num_trial) if trial_infos[i][0] == 1])
    stGO_activity = np.array([aligned_activity[i] for i in range(num_trial) if trial_infos[i][0] == 2])
    stNG_activity = np.array([aligned_activity[i] for i in range(num_trial) if trial_infos[i][0] == 3])
    dtGO_activity = np.array([aligned_activity[i] for i in range(num_trial) if trial_infos[i][0] == 4])
    dtNG_activity = np.array([aligned_activity[i] for i in range(num_trial) if trial_infos[i][0] == 5])

    if neuron_type == 'all':
        neuron_id = np.arange(n_neurons)
    elif neuron_type == 'exc':
        neuron_id = np.arange(int(n_neurons*0.8))
    elif neuron_type == 'inh':
        neuron_id = np.arange(int(n_neurons*0.8),n_neurons)

    trialAvg_stLEVER_activity = np.mean(stLEVER_activity[:,:,neuron_id],axis=0)
    trialAvg_stGO_activity = np.mean(stGO_activity[:,:,neuron_id],axis=0)
    trialAvg_stNG_activity = np.mean(stNG_activity[:,:,neuron_id],axis=0)
    trialAvg_dtGO_activity = np.mean(dtGO_activity[:,:,neuron_id],axis=0)
    trialAvg_dtNG_activity = np.mean(dtNG_activity[:,:,neuron_id],axis=0)

    if normalize:  # normalize activity by maximum value for each neuron in trial average
        trialAvg = np.concatenate([trialAvg_stLEVER_activity, trialAvg_stGO_activity, trialAvg_stNG_activity, trialAvg_dtGO_activity, trialAvg_dtNG_activity], axis=0)
        trialAvg_stLEVER_activity = trialAvg_stLEVER_activity / (np.max(trialAvg, axis=0) + 1e-6)
        trialAvg_stGO_activity = trialAvg_stGO_activity / (np.max(trialAvg, axis=0) + 1e-6)
        trialAvg_stNG_activity = trialAvg_stNG_activity / (np.max(trialAvg, axis=0) + 1e-6)
        trialAvg_dtGO_activity = trialAvg_dtGO_activity / (np.max(trialAvg, axis=0) + 1e-6)
        trialAvg_dtNG_activity = trialAvg_dtNG_activity / (np.max(trialAvg, axis=0) + 1e-6)

    if sort_task is not None:  # sort neural activity
        if sort_task == 'self':
            sort_idx = None
        elif sort_task == 'stLEVER':
            sort_idx = sort_activity(trialAvg_stLEVER_activity)[1]
        elif sort_task == 'stGO':
            sort_idx = sort_activity(trialAvg_stGO_activity)[1]
        elif sort_task == 'stNG':
            sort_idx = sort_activity(trialAvg_stNG_activity)[1]
        elif sort_task == 'dtGO':
            sort_idx = sort_activity(trialAvg_dtGO_activity)[1]
        elif sort_task == 'dtNG':
            sort_idx = sort_activity(trialAvg_dtNG_activity)[1]

        trialAvg_stLEVER_activity, _ = sort_activity(trialAvg_stLEVER_activity,sort_idx)
        trialAvg_stGO_activity, _ = sort_activity(trialAvg_stGO_activity,sort_idx)
        trialAvg_stNG_activity, _ = sort_activity(trialAvg_stNG_activity,sort_idx)
        trialAvg_dtGO_activity, _ = sort_activity(trialAvg_dtGO_activity,sort_idx)
        trialAvg_dtNG_activity, _ = sort_activity(trialAvg_dtNG_activity,sort_idx)

    _, ax = plt.subplots(1,5,figsize=(10, 2.5), sharex=True, sharey=True)
    ax[0].imshow(trialAvg_stLEVER_activity.T, extent=[t[0], t[-1], 0, n_neurons], aspect='auto', vmin=0, vmax=0.8)
    ax[1].imshow(trialAvg_stGO_activity.T, extent=[t[0], t[-1], 0, n_neurons], aspect='auto', vmin=0, vmax=0.8)
    ax[2].imshow(trialAvg_stNG_activity.T, extent=[t[0], t[-1], 0, n_neurons], aspect='auto', vmin=0, vmax=0.8)
    ax[3].imshow(trialAvg_dtGO_activity.T, extent=[t[0], t[-1], 0, n_neurons], aspect='auto', vmin=0, vmax=0.8)
    ax[4].imshow(trialAvg_dtNG_activity.T, extent=[t[0], t[-1], 0, n_neurons], aspect='auto', vmin=0, vmax=0.8)

    for l in [0,3,4]:
        ax[l].vlines(0, 0, n_neurons-1, 'w', '--')
        ax[l].vlines(5, 0, n_neurons-1, 'w', '--')

    for g in [1,2,3,4]:
        ax[g].vlines(2, 0, n_neurons-1, 'w', '--')
        ax[g].vlines(3, 0, n_neurons-1, 'w', '--')

    ax[0].set_title('ST-Lever')
    ax[1].set_title('ST-Go')
    ax[2].set_title('ST-No-Go')
    ax[3].set_title('DT-Go')
    ax[4].set_title('DT-No-Go')

    ax[0].set_ylabel('Trial-averaged activity')

    plt.tight_layout()
    plt.show()

    return aligned_activity


def perf_trials(outputs, task_onset, trial_type, dt, time_steps = 160, batch_size = 16, output_size = 2,
                T_delay=500, T_lever=4500, T_bet=1500, T_cue=1000, T_lick=200):
    """
    Evaluates trial performance for single and dual-task conditions based on output predictions.

    Parameters:
        outputs (torch.Tensor): Model output tensor of shape (time_steps x batch_size x output_size).
        task_onset (torch.Tensor or list): List or tensor of onset indices for each trial.
        trial_type (torch.Tensor or list): List or tensor indicating trial type for each trial.
        dt (float): Time step size in milliseconds.
        time_steps (int): Number of time steps per trial.
        batch_size (int): Number of trials in the batch.
        output_size (int): Number of output channels (e.g., lever and GNG).
        T_delay (int): Delay period in milliseconds.
        T_lever (int): Lever period in milliseconds.
        T_bet (int): Bet period in milliseconds.
        T_cue (int): Cue period in milliseconds.
        T_lick (int): Lick period in milliseconds.

    Returns:
        st_motorSuc_per (float): Percentage of successful single-task lever trials.
        st_hit_per (float): Percentage of hit single-task cognitive trials.
        st_cr_per (float): Percentage of correct rejection single-task cognitive trials.
        dt_motorSuc_per (float): Percentage of successful dual-task lever trials.
        dt_hit_per (float): Percentage of hit dual-task cognitive trials.
        dt_cr_per (float): Percentage of correct rejection dual-task cognitive trials.
    """
    outputs = outputs.reshape(time_steps, batch_size, output_size)

    # Time constants (in steps)
    T_delay_steps = round(T_delay / dt)
    T_tol_steps = round(500 / dt)  # 0.5s tolerance for task onset and offset
    T_lever_steps = round(T_lever / dt)
    T_bet_steps = round(T_bet / dt)
    T_cue_steps = round(T_cue / dt)
    T_lick_steps = round(T_lick / dt)

    # Helper functions for extracting activity windows
    def get_motor_act(i):
        start = int(task_onset[i].item() + T_delay_steps + T_bet_steps)
        end = int(task_onset[i].item() + T_delay_steps + T_lever_steps - T_tol_steps)
        return outputs[start:end, i, 0]

    def get_cog_act(i):
        start = int(task_onset[i].item() + T_delay_steps + T_bet_steps + T_cue_steps)
        end = int(start + T_lick_steps)
        return outputs[start:end, i, 1]

    st_motorSuc = st_hit = st_cr = 0
    dt_motorSuc = dt_hit = dt_cr = 0
    for i, trial in enumerate(trial_type):
        if trial == 1:  # ST-Lever
            if torch.all(get_motor_act(i) > 0.3):
                st_motorSuc += 1
        elif trial == 2:  # ST-Hit
            if torch.any(get_cog_act(i) > 0.7):
                st_hit += 1
        elif trial == 3:  # ST-CR
            if torch.all(get_cog_act(i) < 0.3):
                st_cr += 1
        elif trial in (4, 5):  # DT trials
            motor_act = get_motor_act(i)
            cog_act = get_cog_act(i)
            if torch.all(motor_act > 0.3):
                dt_motorSuc += 1
            if trial == 4 and torch.any(cog_act > 0.7):
                dt_hit += 1
            elif trial == 5 and torch.all(cog_act < 0.3):
                dt_cr += 1

    # Calculate percentages of successes for ST and DT trials
    st_motor = torch.sum(trial_type == 1).item()
    st_motorSuc_per = st_motorSuc / st_motor * 100 
 
    st_cog = torch.sum(trial_type == 2).item() + torch.sum(trial_type == 3).item()
    st_hit_per = st_hit / st_cog * 100 
    st_cr_per = st_cr / st_cog * 100

    dt_total = torch.sum(trial_type == 4).item() + torch.sum(trial_type == 5).item()
    dt_motorSuc_per = dt_motorSuc / dt_total * 100 
    dt_hit_per = dt_hit / dt_total * 100
    dt_cr_per = dt_cr / dt_total * 100

    return st_motorSuc_per, st_hit_per, st_cr_per, dt_motorSuc_per, dt_hit_per, dt_cr_per