"""Read-only acceptance tests for the completed frozen negative control."""
import unittest
import numpy as np
import pandas as pd
import run_negative_control as audit


class NegativeControlAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = audit.read_json(audit.HERE/'config_v1.json')
        cls.check = audit.read_json(audit.HERE/'implementation_check.json')

    def test_01_preregistered_seed_list_and_settings(self):
        seeds = audit.read_frame(audit.HERE/'permutation_seeds.csv')
        self.assertEqual(seeds.permutation_seed.tolist(), list(range(2026090901,2026091000)))
        self.assertEqual(self.cfg['library_sizes'], [10,18,33,60,109,198])
        self.assertEqual(self.cfg['frozen_GCCM_protocol']['convergence'],
            {'mean_curve':'max(0,mean(raw rho))','gain_gt':0,'spearman_gte':.6,
             'pearson_two_sided_p_lt':.05,'fisher_z_95_ci_lower_gt':0})
        self.assertEqual(audit.sha(audit.HERE/'permutation_seeds.csv'),self.cfg['seed_list_sha256'])

    def test_02_original_reproduces_and_sources_unchanged(self):
        self.assertTrue(all(self.check['observed_reproduction'].values()))
        for path,h in self.cfg['source_hashes'].items(): self.assertEqual(audit.sha(path),h)
        for n in audit.RESULT_FILES:
            self.assertEqual(audit.sha(audit.HERE/'observed_reproduction'/n),audit.sha(audit.ORIGINAL/n))

    def test_03_all_99_inputs_are_effect_only_permutations(self):
        for pid,seed in enumerate(audit.SEEDS,1):
            check=audit.validate_input(audit.HERE/'permutations'/f'perm_{pid:03d}',pid,seed)
            self.assertEqual(check['status'],'PASS')

    def test_04_diagnostics_recomputed_from_raw_curves(self):
        for pid in range(1,100):
            self.assertEqual(audit.validate_output(audit.HERE/'permutations'/f'perm_{pid:03d}')['status'],'PASS')

    def test_05_exact_repeat(self):
        for n in ['restricted_input.csv']+audit.RESULT_FILES:
            self.assertEqual(audit.sha(audit.HERE/'repeat_perm_001'/n),audit.sha(audit.HERE/'permutations/perm_001'/n))

    def test_06_aggregate_counts_and_tail_formula(self):
        d=audit.read_frame(audit.HERE/'permutation_diagnostics.csv')
        curves=audit.read_frame(audit.HERE/'permutation_library_statistics.csv')
        self.assertEqual(len(d),594);self.assertEqual(len(curves),3564)
        self.assertEqual(d.groupby('permutation_id').size().tolist(),[6]*99)
        tail=audit.read_frame(audit.HERE/'spatial_negative_control_tails.csv')
        for r in tail.itertuples():
            vals=d.loc[d.variable==r.variable,r.statistic].dropna()
            self.assertAlmostEqual(r.negative_control_tail_probability,(1+(vals>=r.observed).sum())/(1+len(vals)),places=14)

    def test_07_label_boundary_and_no_comparison(self):
        for key in ['tuning_calibration_labels_accessed','validation_labels_accessed',
                    'final_calibration_labels_accessed','test_labels_accessed',
                    'oracle_files_read_or_hashed','predictor_retrained','primary_validation_comparison_executed']:
            self.assertIs(self.check[key],False)
        self.assertEqual(self.cfg['scenario'],'S4')
        for p in (audit.HERE/'permutations').glob('*/restricted_input.csv'):
            self.assertEqual(audit.csv_tokens(p)[0],audit.SCHEMA)
        self.assertFalse((audit.HERE/'permutations/perm_100').exists())

    def test_08_manifest_integrity(self):
        for name,digest in audit.read_json(audit.HERE/'artifact_manifest.json').items():
            self.assertEqual(audit.sha(audit.HERE/name),digest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
