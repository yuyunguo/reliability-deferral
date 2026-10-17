"""train_pipeline must split with the patient ids of the data_dir it was given."""
import inspect

from deferral_frontier.src import train_pipeline as tp


def test_train_pipeline_uses_its_data_dir_and_exposes_split_seed():
    src = inspect.getsource(tp.train_pipeline)
    assert "load_patient_split(data_dir" in src
    assert "split_seed" in inspect.signature(tp.train_pipeline).parameters
