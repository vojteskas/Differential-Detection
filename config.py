local_config = {
    "argv": ["--local"],
    "device": "cuda",
    "data_dir": "/mnt/e/VUT/Deepfakes/Datasets/",
    "asvspoof2019la": {
        "subdir": "LA",
        "train_protocol": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev_protocol": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval_protocol": "ASVspoof2019.LA.cm.eval.trl.txt",
    },
    "asvspoof2021la": {
        # need to take train and dev from 2019 and eval from 2021
        "subdir": "",
        "train_protocol": "LA/ASVspoof2019.LA.cm.train.trn.txt",
        "dev_protocol": "LA/ASVspoof2019.LA.cm.dev.trl.txt",
        "eval_protocol": "asvspoof2021/LA/trial_metadata.txt",
    },
    "asvspoof2021df": {
        "subdir": "",
        "train_protocol": "LA/ASVspoof2019.LA.cm.train.trn.txt",
        "dev_protocol": "LA/ASVspoof2019.LA.cm.dev.trl.txt",
        "eval_protocol": "asvspoof2021/DF/trial_metadata.txt",
    },
    "batch_size": 16,
    "num_epochs": 1,
}

metacentrum_config = {
    "argv": ["--metacentrum"],
    "device": "cuda",
    "data_dir": "./",
    "asvspoof2019la": {
        "subdir": "LA19",
        "train_protocol": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev_protocol": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval_protocol": "ASVspoof2019.LA.cm.eval.trl.txt",
    },
    "asvspoof2021la": {
        # need to take train and dev from 2019 and eval from 2021
        "subdir": "LA21",
        "train_protocol": "LA/ASVspoof2019.LA.cm.train.trn.txt",
        "dev_protocol": "LA/ASVspoof2019.LA.cm.dev.trl.txt",
        "eval_protocol": "asvspoof2021/LA/trial_metadata.txt",
    },
    "asvspoof2021df": {
        "subdir": "DF21",
        "train_protocol": "LA/ASVspoof2019.LA.cm.train.trn.txt",
        "dev_protocol": "LA/ASVspoof2019.LA.cm.dev.trl.txt",
        "eval_protocol": "asvspoof2021/DF/trial_metadata.txt",
    },
    "batch_size": 32,
    "num_epochs": 10,
}
