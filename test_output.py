# content of test_sample.py
import numpy as np
import pytest
import yaml

with open("Options.yaml") as f:
    Options = yaml.safe_load(f)

@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)

def test_cluster():
    #Load "correct" results data
    Results = np.load("Tests/Test_results.npz",allow_pickle=True)

    #Rerun clustering
    exec(open("Find_clusters3d.py").read())

    #Open new results    
    formatter =  "{:1.1f}"
    outfile = Options["outdir"] +  Options["str_result"] + formatter.format( Options["distthresh"]) +\
    "_tim_" + formatter.format( Options["timthresh"]) + "_length_" + formatter.format( Options["lngthresh"]) +\
    "_timlength_" + formatter.format( Options["timlngthresh"]*6.0) + "_" + Options["distmeth"] + "_corrected.npz"

    Results_test = np.load(outfile,allow_pickle=True)    

    #Assert if they are the same
    assert all(Results_test["sorted_clusters"] == Results["sorted_clusters"])
    assert all(Results_test["sorted_subclusters_bjerknes"] == Results["sorted_subclusters_length"])
    assert all(Results_test["sorted_subclusters_stagnant"] == Results["sorted_subclusters_nolength"])
