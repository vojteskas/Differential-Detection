#!/usr/bin/env python
# This script generates a bash script for submitting a job to the MetaCentrum PBS system.


class PBSheaders:
    def __init__(
        self,
        jobname: str,  # name of the job
        queue="gpu@pbs-m1.metacentrum.cz",  # queue name
        walltime="24:00:00",  # maximum time the job can run
        nodes=1,  # number of nodes
        cpus=4,  # number of cpus on a node
        mem=200,  # memory per node in GB
        gpus=1,  # number of gpus
        gpu_mem=20,  # minimum gpu memory in GB
        scratch_size=100,  # minimum scratch_dir size in GB, implicit type is scratch_ssd
        email_notification_flags="ae",  # when to send email notifications, see https://docs.metacentrum.cz/computing/email-notif/
    ):
        self.jobname = jobname
        self.queue = queue
        self.walltime = walltime
        self.nodes = nodes
        self.cpus = cpus
        self.mem = mem
        self.gpus = gpus
        self.gpu_mem = gpu_mem
        self.scratch_size = scratch_size
        self.email_notification_flags = email_notification_flags

    def __str__(self):
        header = [
            f"#!/bin/bash",
            f"#PBS -N {self.jobname}",
            f"#PBS -q {self.queue}",
            f"#PBS -l walltime={self.walltime}",
            f"#PBS -l select={self.nodes}:ncpus={self.cpus}:mem={self.mem}gb:ngpus={self.gpus}:gpu_mem={self.gpu_mem}gb:scratch_ssd={self.scratch_size}gb",
            f"#PBS -m {self.email_notification_flags}",
        ]

        return "\n".join(header)

    def __repr__(self):
        return self.__str__()

    def __call__(self):
        return self.__str__()


class Job:
    def __init__(
        self,
        jobname: str,  # name of the job
        project_archive_path="DP/",  # path to the project source archive from home (~) directory
        project_archive_name="dp.zip",  # name of the project source archive
        dataset_archive_path="deepfakes/datasets/",  # path to the dataset from home (~) directory
        dataset_archive_name="LA.zip",  # name of the dataset archive
        checkpoint_file_path=None,  # path to the checkpoint file from
        checkpoint_archive_name=None,  # name of the checkpoint archive
        checkpoint_file_from_archive_name=None,  # name of the checkpoint file from the archive
        execute_list=[
            ("train_and_eval.py", ["--metacentrum"])
        ],  # doubles of script name and list of arguments
        train=True,  # if training, copy training LA19 dataset aswell
        copy_results=True,  # copy results back to home directory
    ):
        self.jobname = jobname
        self.project_archive_path = project_archive_path
        self.project_archive_name = project_archive_name
        self.dataset_archive_path = dataset_archive_path
        self.dataset_archive_name = dataset_archive_name
        self.checkpoint_file_path = checkpoint_file_path
        self.checkpoint_archive_name = checkpoint_archive_name
        self.checkpoint_file_from_archive_name = checkpoint_file_from_archive_name
        self.execute_list = execute_list
        self.train = train
        self.copy_results = copy_results

    def __str__(self):
        env_script = [
            "export OMP_NUM_THREADS=$PBS_NUM_PPN",  # set the number of threads to the number of cpus
            "\n",
            # variable declaration
            f"name={self.jobname}",
            f'archivename="$name"_Results.zip',
            "DATADIR=/storage/brno2/home/vojteskas",
            "\n",
            # create tmpdir in scratch for caching (pip) etc.
            'cd "$SCRATCHDIR" || exit 1',
            "mkdir TMPDIR",
            'export TMPDIR="$SCRATCHDIR/TMPDIR"',
            "\n",
            # environment setup
            'echo "Creating conda environment"',
            "module add gcc",
            "module add conda-modules-py37",
            'conda create --prefix "$TMPDIR/condaenv" python=3.10 -y >/dev/null 2>&1',
            'conda activate "$TMPDIR/condaenv" >/dev/null 2>&1',
            "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y >/dev/null 2>&1",
            "\n",
            # copy project files
            'echo "Copying project files"',
            f"cp $DATADIR/{self.project_archive_path}{self.project_archive_name} .",  # copy to scratchdir
            f"unzip {self.project_archive_name} >/dev/null 2>&1",
            "\n",
            # install project requirements
            'echo "Installing project requirements"',
            'pip install -r requirements.txt --cache-dir "$TMPDIR" >/dev/null 2>&1',
            "\n",
        ]

        data_script = [
            # copy dataset
            # TODO: Allow for multiple datasets to be copied
            'echo "Copying dataset(s)"',
            f"cp -r $DATADIR/{self.dataset_archive_path}{self.dataset_archive_name} .",  # copy to scratchdir
            f"tar -xzf {self.dataset_archive_name} >/dev/null 2>&1",
            "\n",
        ]
        # copy 2019 dataset (training data) aswell if 2021 or InTheWild datasets are used
        if self.train and ("21" in self.dataset_archive_name or "InTheWild" in self.dataset_archive_name):
            data_script.extend(
                [
                    # copy 2019 dataset
                    f"cp -r $DATADIR/{self.dataset_archive_path}LA19.tar.gz .",  # copy to scratchdir
                    f"tar -xzf LA19.tar.gz >/dev/null 2>&1",
                    "\n",
                ]
            )

        checkpoint_script = []
        if self.checkpoint_archive_name:
            checkpoint_script = [
                # copy checkpoint
                'echo "Copying checkpoint archive"',
                f"cp $DATADIR/DP/{self.checkpoint_archive_name} .",  # copy to scratchdir
                f"unzip {self.checkpoint_archive_name} {self.checkpoint_file_from_archive_name} >/dev/null 2>&1",
                "\n",
            ]
        if self.checkpoint_file_path:
            checkpoint_script = [
                # copy checkpoint
                'echo "Copying checkpoint file"',
                f"cp $DATADIR/DP/{self.checkpoint_file_path} .",  # copy to scratchdir
                "\n",
            ]

        exec_script = [
            # run the script
            "chmod 755 ./*.py",
            'echo "Running the script"',
        ]
        for script, args in self.execute_list:
            exec_script.append(f"./{script} {' '.join(args)} 2>&1\n")

        results_script = []
        if self.copy_results:
            results_script = [
                # copy results
                'echo "Copying results"',
                'find . -type d -name "__pycache__" -exec rm -rf {} +',  # remove __pycache__ directories
                'zip -r "$archivename" ./*.png ./*.pt ./*.txt >/dev/null 2>&1',
                f'cp "$archivename" $DATADIR/{self.project_archive_path}$archivename >/dev/null 2>&1',
                "\n",
            ]

        cleanup_script = [
            # cleanup
            "clean_scratch"
        ]

        return "\n".join(
            env_script + data_script + checkpoint_script + exec_script + results_script + cleanup_script
        )

    def __repr__(self):
        return self.__str__()

    def __call__(self):
        return self.__str__()


def generate_job_script(
    jobname: str,  # name of the job
    file_name=None,  # name of the file to save the script to, if None, the script is not saved
    **kwargs,  # keyword arguments for PBSheaders and Job
):
    pbsheaders_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key
        in [
            "queue",
            "walltime",
            "nodes",
            "cpus",
            "mem",
            "gpus",
            "gpu_mem",
            "scratch_size",
            "email_notification_flags",
        ]
    }
    job_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key
        in [
            "project_archive_path",
            "project_archive_name",
            "dataset_archive_path",
            "dataset_archive_name",
            "checkpoint_file_path",
            "checkpoint_archive_name",
            "checkpoint_file_from_archive_name",
            "execute_list",
            "train",
            "copy_results",
        ]
    }
    pbs = PBSheaders(jobname, **pbsheaders_kwargs)
    job = Job(jobname, **job_kwargs)
    script = f"{pbs}\n\n{job}"

    if file_name:
        with open(file_name, "w", newline="\n") as file:
            file.write(script)

    return script


if __name__ == "__main__":
    # # Modify parameters and arguments here
    # dataset = "ASVspoof2019LADataset_pair"
    # dshort = "LA19"
    # for c, ep in [
    #     ("FFDiff", 20),
    #     ("FFDiffAbs", 15),
    #     ("FFDiffQuadratic", 15),
    #     ("FFConcat1", 15),
    #     ("FFConcat3", 10),
    #     ("FFLSTM", 10),
    # ]:
    #     command = [
    #         (
    #             "eval.py",
    #             [
    #                 "--metacentrum",
    #                 "--dataset",
    #                 dataset,
    #                 "--classifier",
    #                 f"{c}",
    #                 "--extractor",
    #                 "XLSR_300M",
    #                 "--processor",
    #                 "MHFA",
    #                 "--checkpoint",
    #                 f"{c}_{ep}.pt",
    #             ],
    #         )
    #     ]
    #     generate_job_script(
    #         jobname=f"EVAL_{c}_{dshort}_{ep}",
    #         file_name=f"scripts/{c}_{dshort}_{ep}.sh",
    #         project_archive_name="dp.zip",
    #         dataset_archive_name=f"{dshort}.tar.gz",
    #         checkpoint_archive_name=f"NEW_{c}_LA19_Results.zip",
    #         checkpoint_file_from_archive_name=f"{c}_{ep}.pt",
    #         execute_list=command,
    #         train=False,
    #     )
    c = "FF"
    dshort = "ASVspoof5"
    for dataset in ("ASVspoof5Dataset_single", "ASVspoof5Dataset_single_augmented"):
        for extractor in ("WavLM_base", "HuBERT_base"):
            command = [
                (
                    "train_and_eval.py",
                    [
                        "--metacentrum",
                        "--dataset",
                        dataset,
                        "--classifier",
                        f"{c}",
                        "--extractor",
                        f"{extractor}",
                        "--processor",
                        "MHFA",
                        "--num_epochs",
                        "20",
                    ],
                )
            ]
            generate_job_script(
                jobname=f"DP_{extractor}_{c}_{dshort + 'aug' if 'aug' in dataset else dshort}",
                file_name=f"scripts/{c}_{dshort}_{dataset}_{extractor}.sh",
                project_archive_name="dp.zip",
                dataset_archive_name=f"{dshort}.tar.gz",
                execute_list=command,
            )
