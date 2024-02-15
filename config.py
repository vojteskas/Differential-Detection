local_config = {
    "argv": ["--local"],
    "device": "cuda",
    "data_dir": "/mnt/e/VUT/Deepfakes/Datasets/LA",
    "train_protocol": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev_protocol": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval_protocol": "ASVspoof2019.LA.cm.eval.trl.txt",
    "batch_size": 16,
    "num_epochs": 1,
}

metacentrum_config = {
    "argv": ["--metacentrum"],
    "device": "cuda",
    "data_dir": "./LA",
    "train_protocol": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev_protocol": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval_protocol": "ASVspoof2019.LA.cm.eval.trl.txt",
    "batch_size": 32,  # 64 needs more then 16GB of GPU memory, 32 should be fine
    "num_epochs": 10,
}
