import numpy as np
import matplotlib.pyplot as plt

def dataset(dt, batch_size=16, DT = False, DTgo = None):
 
    T = 8000
    T_cue = 1000
    T_lick = 200
    T_lever = 4500
    T_delay = 500
    T_bet = 1500

    all_inputs = []
    all_targets = []
    all_trial_type = []

    cog_tuning, motor_tuning, rule_tuning = create_tuning_functions()
    
    for _ in range(batch_size):
        t_step_start = int(np.random.uniform(1000 / dt, (T - T_lever - T_delay - 1000) / dt))
        # single_input = np.zeros((round(T / dt), 42)).astype(np.float32)
        single_input = np.random.normal(0, np.sqrt(2/0.5)*0.1, size=(round(T / dt), 42)).astype(np.float32)
        single_target = np.zeros((round(T / dt), 2)).astype(np.float32)

        # Randomly pick a task: 1: ST-Lever, 2: ST-Go, 3: ST-No-Go, 4: DT-Go, 5: DT-No-Go
        if DT:
            trial_type = np.random.choice([1, 2, 3, 4, 5])
        else:
            trial_type = np.random.choice([1, 2, 3])

        LEVER = None
        GNG = None
        
        if trial_type == 1 or trial_type == 4 or trial_type == 5:
            LEVER = 1
            single_input += rule_tuning[:,0]    # Context-Lever
        if trial_type == 2 or trial_type == 4:
            GNG = 1
            single_input += rule_tuning[:,1]   # Context-GNG
        if trial_type == 3 or trial_type == 5:
            GNG = 2
            single_input += rule_tuning[:,1]   # Context-GNG
        if trial_type == 4 or trial_type == 5:
            single_input += rule_tuning[:,2]    # Context-DT

        if LEVER == 1:
            single_input[t_step_start : t_step_start + round(T_lever / dt), :] += motor_tuning[:,0] 
            if GNG == 1 and DTgo is not None:
                single_target[t_step_start - round(1000/dt) : t_step_start + round(6000/dt),0] = DTgo
            else:
                single_target[t_step_start + round(T_delay / dt) : t_step_start + round((T_delay + T_lever) / dt), 0] = 1
        
        if GNG == 1:
            single_input[t_step_start + round((T_delay+T_bet) / dt):t_step_start + round((T_delay+T_bet+T_cue) / dt), :] += cog_tuning[:,0]
            single_target[t_step_start + round((T_delay+T_bet+T_cue) / dt):t_step_start + round((T_delay+T_bet+T_cue+T_lick) / dt),1] = 1

        if GNG == 2:
            single_input[t_step_start + round((T_delay+T_bet) / dt):t_step_start + round((T_delay+T_bet+T_cue) / dt), :] += cog_tuning[:,1]

        # relu
        single_input = np.maximum(single_input, 0)
        
        all_inputs.append(single_input)
        all_targets.append(single_target)
        all_trial_type.append(trial_type)

    # Stack each trial along batch dimension -> (time, batch, features)
    all_inputs = np.stack(all_inputs, axis=0)  # (batch, time, input_size)
    all_targets = np.stack(all_targets, axis=0)  # (batch, time, output_size)
    all_inputs = np.transpose(all_inputs, (1, 0, 2))  # (time, batch, input_size)
    all_targets = np.transpose(all_targets, (1, 0, 2))  # (time, batch, output_size)

    return all_inputs, all_targets, t_step_start, all_trial_type

def create_tuning_functions(num_motor_stimuli=1, num_motor_tuned=24,num_cog_stimuli=2, num_cog_tuned=12, num_rules=3, num_rule_tuned=6, kappa=2):

    n_input = num_motor_tuned + num_cog_tuned + num_rule_tuned
    motor_tuning = np.zeros((n_input, num_motor_stimuli))
    cog_tuning = np.zeros((n_input, num_cog_stimuli))
    rule_tuning = np.zeros((n_input, num_rules))

    motor_pref = np.float32(np.arange(0,360,360/num_motor_tuned)) # generate list of prefered positions
    motor_stim = np.float32(np.arange(0,360,360/num_motor_stimuli))
    
    cog_pref = np.float32(np.arange(0,360,360/num_cog_tuned)) # generate list of prefered cues
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

    return cog_tuning, motor_tuning, rule_tuning


def plot_behav_per(win,dt,num_trial,trial_infos,performance_dict,task_onsets):
    
    # visualize behavioral performance
    win = np.array(win)*1000/dt
    win = win.astype(int)

    aligned_performance = [performance_dict[i][task_onsets[i]+win[0]:task_onsets[i]+win[1],:] for i in range(num_trial)]

    stLEVER_performance = np.array([aligned_performance[i] for i in range(num_trial) if trial_infos[i] == 1])
    stGO_performance = np.array([aligned_performance[i] for i in range(num_trial) if trial_infos[i] == 2])
    stNG_performance = np.array([aligned_performance[i] for i in range(num_trial) if trial_infos[i] == 3])
    dtGO_performance = np.array([aligned_performance[i] for i in range(num_trial) if trial_infos[i] == 4])
    dtNG_performance = np.array([aligned_performance[i] for i in range(num_trial) if trial_infos[i] == 5])

    t = np.arange(win[0], win[1]) * dt / 1000
    fig, ax = plt.subplots(2,5,figsize=(12, 5), sharex=True, sharey=True)
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

    win = np.array(win)*1000/dt
    win = win.astype(int)
    t = np.arange(win[0], win[1]) * dt / 1000

    aligned_activity = [activity_dict[i][task_onsets[i]+win[0]:task_onsets[i]+win[1],:] for i in range(num_trial)]
    stLEVER_activity = np.array([aligned_activity[i] for i in range(num_trial) if trial_infos[i] == 1])
    stGO_activity = np.array([aligned_activity[i] for i in range(num_trial) if trial_infos[i] == 2])
    stNG_activity = np.array([aligned_activity[i] for i in range(num_trial) if trial_infos[i] == 3])
    dtGO_activity = np.array([aligned_activity[i] for i in range(num_trial) if trial_infos[i] == 4])
    dtNG_activity = np.array([aligned_activity[i] for i in range(num_trial) if trial_infos[i] == 5])

    if neuron_type == 'all':
        neuron_id = np.arange(stLEVER_activity.shape[2])
    elif neuron_type == 'exc':
        neuron_id = np.arange(int(stLEVER_activity.shape[2]*0.8))
    elif neuron_type == 'inh':
        neuron_id = np.arange(int(stLEVER_activity.shape[2]*0.8),stLEVER_activity.shape[2])

    trialAvg_stLEVER_activity = np.mean(stLEVER_activity[:,:,neuron_id],axis=0)
    trialAvg_stGO_activity = np.mean(stGO_activity[:,:,neuron_id],axis=0)
    trialAvg_stNG_activity = np.mean(stNG_activity[:,:,neuron_id],axis=0)
    trialAvg_dtGO_activity = np.mean(dtGO_activity[:,:,neuron_id],axis=0)
    trialAvg_dtNG_activity = np.mean(dtNG_activity[:,:,neuron_id],axis=0)

    if normalize:
        # concatenate all task conditions
        trialAvg = np.concatenate([trialAvg_stLEVER_activity, trialAvg_stGO_activity, trialAvg_stNG_activity, trialAvg_dtGO_activity, trialAvg_dtNG_activity], axis=0)
        # normalize activity by maximum value for each neuron in trial average
        trialAvg_stLEVER_activity = trialAvg_stLEVER_activity / (np.max(trialAvg, axis=0) + 1e-6)
        trialAvg_stGO_activity = trialAvg_stGO_activity / (np.max(trialAvg, axis=0) + 1e-6)
        trialAvg_stNG_activity = trialAvg_stNG_activity / (np.max(trialAvg, axis=0) + 1e-6)
        trialAvg_dtGO_activity = trialAvg_dtGO_activity / (np.max(trialAvg, axis=0) + 1e-6)
        trialAvg_dtNG_activity = trialAvg_dtNG_activity / (np.max(trialAvg, axis=0) + 1e-6)

    hidden_size = len(neuron_id)

    # sort neural activity
    if sort_task is not None:
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

    fig, ax = plt.subplots(1,5,figsize=(10, 2.5), sharex=True, sharey=True)
    ax[0].imshow(trialAvg_stLEVER_activity.T, extent=[t[0], t[-1], 0, hidden_size], aspect='auto', vmin=0, vmax=0.8)
    ax[1].imshow(trialAvg_stGO_activity.T, extent=[t[0], t[-1], 0, hidden_size], aspect='auto', vmin=0, vmax=0.8)
    ax[2].imshow(trialAvg_stNG_activity.T, extent=[t[0], t[-1], 0, hidden_size], aspect='auto', vmin=0, vmax=0.8)
    ax[3].imshow(trialAvg_dtGO_activity.T, extent=[t[0], t[-1], 0, hidden_size], aspect='auto', vmin=0, vmax=0.8)
    ax[4].imshow(trialAvg_dtNG_activity.T, extent=[t[0], t[-1], 0, hidden_size], aspect='auto', vmin=0, vmax=0.8)

    for l in [0,3,4]:
        ax[l].vlines(0, 0, hidden_size-1, 'w', '--')
        ax[l].vlines(5, 0, hidden_size-1, 'w', '--')

    for g in [1,2,3,4]:
        ax[g].vlines(2, 0, hidden_size-1, 'w', '--')
        ax[g].vlines(3, 0, hidden_size-1, 'w', '--')

    ax[0].set_title('ST-Lever')
    ax[1].set_title('ST-Go')
    ax[2].set_title('ST-No-Go')
    ax[3].set_title('DT-Go')
    ax[4].set_title('DT-No-Go')

    ax[0].set_ylabel('Trial-averaged activity')

    plt.tight_layout()
    plt.show()

    return aligned_activity