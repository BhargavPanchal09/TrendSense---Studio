import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from v2_engine.content_model import ContentRegressor, ContentEnsemble, load_content_model
from v2_engine.benchmark_content import split_indices


class ContentTests(unittest.TestCase):
    def test_saved_preprocessing_and_predictions_match(self):
        torch.manual_seed(42)
        model = ContentRegressor().eval()
        model.feature_mean.fill_(.5)
        model.feature_scale.fill_(2.)
        x = torch.randn(4, 385)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            torch.save({'format_version': 1, 'input_features': 385,
                        'config': {'width': 128, 'dropout': .35}, 'states': [model.state_dict()]}, path)
            loaded = load_content_model(path)
            with torch.no_grad():
                torch.testing.assert_close(loaded(x), model(x))
                self.assertTrue(((loaded(x) >= 0) & (loaded(x) <= 100)).all())

    def test_exact_duplicates_do_not_cross_splits(self):
        x = np.random.default_rng(42).normal(size=(100, 385)).astype('float32')
        x[0] = x[1]
        splits = split_indices(x)
        keys = [{x[i].tobytes() for i in indices} for indices in splits]
        self.assertFalse(keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
        self.assertEqual(sum(map(len, splits)), 99)

    def test_ensemble_rejects_missing_members(self):
        with self.assertRaises(ValueError):
            ContentEnsemble({'width': 128, 'dropout': .35}, [])


if __name__ == '__main__':
    unittest.main()
