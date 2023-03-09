import json
import os
import unittest, pandas as pd
from cadCAD.configuration import Experiment
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from testing.models import param_sweep, policy_aggregation


class RowCountTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        exp = Experiment()
        self.sys_model_A_id = "sys_model_A"
        exp.append_model(
            model_id=self.sys_model_A_id,
            sim_configs=param_sweep.sim_config,
            initial_state=param_sweep.genesis_states,
            env_processes=param_sweep.env_process,
            partial_state_update_blocks=param_sweep.partial_state_update_blocks
        )
        self.sys_model_B_id = "sys_model_B"
        exp.append_model(
            model_id=self.sys_model_B_id,
            sim_configs=param_sweep.sim_config,
            initial_state=param_sweep.genesis_states,
            env_processes=param_sweep.env_process,
            partial_state_update_blocks=param_sweep.partial_state_update_blocks
        )
        self.sys_model_C_id = "sys_model_C"
        exp.append_model(
            model_id=self.sys_model_C_id,
            sim_configs=policy_aggregation.sim_config,
            initial_state=policy_aggregation.genesis_states,
            partial_state_update_blocks=policy_aggregation.partial_state_update_block,
            policy_ops=[lambda a, b: a + b, lambda y: y * 2] # Default: lambda a, b: a + b
        )

        simulation = 3
        model_A_sweeps = len(param_sweep.sim_config)
        model_B_sweeps = len(param_sweep.sim_config)
        model_C_sweeps = 1
        # total_sweeps = model_A_sweeps + model_B_sweeps

        model_A_runs = param_sweep.sim_config[0]['N']
        model_B_runs = param_sweep.sim_config[0]['N']
        model_C_runs = policy_aggregation.sim_config['N']
        # total_runs = model_A_runs + model_B_runs

        model_A_timesteps = len(param_sweep.sim_config[0]['T'])
        model_B_timesteps = len(param_sweep.sim_config[0]['T'])
        model_C_timesteps = len(policy_aggregation.sim_config['T'])

        model_A_substeps = len(param_sweep.partial_state_update_blocks)
        model_B_substeps = len(param_sweep.partial_state_update_blocks)
        model_C_substeps = len(policy_aggregation.partial_state_update_block)
        # total_substeps = model_A_substeps + model_B_substeps

        model_A_init_rows = model_A_runs * model_A_sweeps
        model_B_init_rows = model_B_runs * model_B_sweeps
        model_C_init_rows = model_C_runs * 1
        self.model_A_rows = model_A_init_rows + (model_A_sweeps * (model_A_runs * model_A_timesteps * model_A_substeps))
        self.model_B_rows = model_B_init_rows + (model_B_sweeps * (model_B_runs * model_B_timesteps * model_B_substeps))
        self.model_C_rows = model_C_init_rows + (model_C_sweeps * (model_C_runs * model_C_timesteps * model_C_substeps))


        exec_mode = ExecutionMode()
        local_mode_ctx = ExecutionContext(context=exec_mode.local_mode)
        simulation = Executor(exec_context=local_mode_ctx, configs=exp.configs)
        raw_results, _, _ = simulation.execute()

        results_df = pd.DataFrame(raw_results)
        param_sweep_df = pd.read_pickle("expected_results/param_sweep_4.pkl")
        policy_agg_df = pd.read_pickle("expected_results/policy_agg_4.pkl")
        self.param_sweep_df_rows = len(param_sweep_df.index)
        self.policy_agg_df_rows = len(policy_agg_df.index)

        self.expected_rows = self.param_sweep_df_rows + self.param_sweep_df_rows + self.policy_agg_df_rows
        self.expected_rows_from_api = self.model_A_rows + self.model_B_rows + self.model_C_rows
        self.result_rows = len(results_df.index)

    def test_row_count(self):
        equal_row_count = self.expected_rows == self.expected_rows_from_api == self.result_rows
        self.assertEqual(equal_row_count, True, "Row Count Mismatch between Expected and Multi-Model simulation results")
    def test_row_count_from_api(self):
        self.assertEqual(self.expected_rows == self.expected_rows_from_api, True, "API not producing Expected simulation results")
    def test_row_count_from_results(self):
        self.assertEqual(self.expected_rows == self.result_rows, True, "Engine not producing Expected simulation results")
    def test_row_count_from_sys_model_A(self):
        self.assertEqual(self.model_A_rows == self.param_sweep_df_rows, True, f"{self.sys_model_A_id}: Row Count Mismatch with Expected results")
    def test_row_count_from_sys_model_B(self):
        self.assertEqual(self.model_B_rows == self.param_sweep_df_rows, True, f"{self.sys_model_B_id}: Row Count Mismatch with Expected results")
    def test_row_count_from_sys_model_C(self):
        self.assertEqual(self.model_C_rows == self.policy_agg_df_rows, True, f"{self.sys_model_C_id}: Row Count Mismatch with Expected results")
    def test_a_b_row_count(self):
        file_path = f'{os.getcwd()}/testing/tests/a_b_tests/0_4_23_record_count.json'
        record_count_0_4_23 = json.load(open(file_path))['record_count']
        record_count_current = self.result_rows
        self.assertEqual(record_count_current > record_count_0_4_23, True, "Invalid Row Count for current version")


if __name__ == '__main__':
    unittest.main()
