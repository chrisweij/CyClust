# content of test_sample.py
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)

def test_cluster():
    #Load "correct" results data
    Results = np.load("Tests/Test_results.npz",allow_pickle=True)

    #Rerun clustering
    exec(open("Find_clusters3d.py").read())

    #Open new results
    Results_test = np.load("Clusters_output/Winter_2011_2012_EI_1.0_tim_36.0_length_1.5_timlength_48.0_AlongTracksDirect_corrected.npz",allow_pickle=True)    

    #Assert if they are the same
    assert all(Results_test["sorted_clusters"] == Results["sorted_clusters"])
    assert all(Results_test["sorted_subclusters_bjerknes"] == Results["sorted_subclusters_length"])
    assert all(Results_test["sorted_subclusters_stagnant"] == Results["sorted_subclusters_nolength"])
