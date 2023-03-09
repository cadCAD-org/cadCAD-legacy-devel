from pathos.multiprocessing import ProcessPool
from joblib import Parallel, delayed
from collections import Counter

from cadCAD.utils import flatten

VarDictType = dict[str, list[object]]
StatesListsType = list[dict[str, object]]
ConfigsType = list[tuple[list[callable], list[callable]]]
EnvProcessesType = dict[str, callable]


def single_proc_exec(
    simulation_execs: list[callable],
    var_dict_list: list[VarDictType],
    states_lists: list[StatesListsType],
    configs_structs: list[ConfigsType],
    env_processes_list: list[EnvProcessesType],
    Ts: list[range],
    SimIDs,
    Ns: list[int],
    ExpIDs: list[int],
    SubsetIDs,
    SubsetWindows,
    configured_n
):
    print(f'Execution Mode: single_threaded')
    params = [
        simulation_execs, states_lists, configs_structs, env_processes_list,
        Ts, SimIDs, Ns, SubsetIDs, SubsetWindows
    ]
    simulation_exec, states_list, config, env_processes, T, sim_id, N, subset_id, subset_window = list(
        map(lambda x: x.pop(), params)
    )
    result = simulation_exec(
        var_dict_list, states_list, config, env_processes, T, sim_id, N, subset_id, subset_window, configured_n
    )
    return flatten(result)


def parallelize_simulations(
    simulation_execs: list[callable],
    var_dict_list: list[VarDictType],
    states_lists: list[StatesListsType],
    configs_structs: list[ConfigsType],
    env_processes_list: list[EnvProcessesType],
    Ts: list[range],
    SimIDs,
    Ns: list[int],
    ExpIDs: list[int],
    SubsetIDs: list[int],
    SubsetWindows,
    configured_n
):

    print(f'Execution Mode: parallelized')
    params = list(
        zip(
            simulation_execs, var_dict_list, states_lists, configs_structs, env_processes_list,
            Ts, SimIDs, Ns, SubsetIDs, SubsetWindows
        )
    )

    len_configs_structs = len(configs_structs)

    unique_runs = Counter(SimIDs)
    sim_count = max(unique_runs.values())
    highest_divisor = int(len_configs_structs / sim_count)

    new_configs_structs, new_params = [], []
    for count in range(len(params)):
        if count == 0:
            new_params.append(
                params[count: highest_divisor]
            )
            new_configs_structs.append(
                configs_structs[count: highest_divisor]
            )
        elif count > 0:
            new_params.append(
                params[count * highest_divisor: (count + 1) * highest_divisor]
            )
            new_configs_structs.append(
                configs_structs[count * highest_divisor: (count + 1) * highest_divisor]
            )

    def process_executor(params):
        if len_configs_structs > 1:
            with ProcessPool(processes=len_configs_structs) as pp:
                results = pp.map(
                    lambda t: t[0](t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], configured_n), params
                )
        else:
            t = params[0]
            results = t[0](t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], configured_n)
        return results

    results = flatten(list(map(lambda params: process_executor(params), new_params)))

    return results


def local_simulations(
        simulation_execs: list[callable],
        var_dict_list: list[VarDictType],
        states_lists: list[StatesListsType],
        configs_structs: list[ConfigsType],
        env_processes_list: list[EnvProcessesType],
        Ts: list[range],
        SimIDs,
        Ns: list[int],
        ExpIDs: list[int],
        SubsetIDs: list[int],
        SubsetWindows,
        configured_n
    ):
    config_amt = len(configs_structs)

    _params = None
    if config_amt == 1: # and configured_n != 1
        _params = var_dict_list[0]
        return single_proc_exec(
            simulation_execs, _params, states_lists, configs_structs, env_processes_list,
            Ts, SimIDs, Ns, ExpIDs, SubsetIDs, SubsetWindows, configured_n
        )
    elif config_amt > 1: # and configured_n != 1
        _params = var_dict_list
        return parallelize_simulations(
            simulation_execs, _params, states_lists, configs_structs, env_processes_list,
            Ts, SimIDs, Ns, ExpIDs, SubsetIDs, SubsetWindows, configured_n
        )
        # elif config_amt > 1 and configured_n == 1:
