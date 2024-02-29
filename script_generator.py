#!/usr/bin/env python

class PBSheaders:
    def __init__(
        self,
        jobname: str,  # name of the job
        queue="gpu@meta-pbs.metacentrum.cz",  # queue name
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
        checkpoint_archive_name=None,  # name of the checkpoint archive
        executable_script="train_and_eval.py",  # name of the script to be executed
        executable_script_args=["--metacentrum"],  # arguments for the script
    ):
        self.jobname = jobname
        self.project_archive_path = project_archive_path
        self.project_archive_name = project_archive_name
        self.dataset_archive_path = dataset_archive_path
        self.dataset_archive_name = dataset_archive_name
        self.checkpoint_archive_name = checkpoint_archive_name
        self.executable_script = executable_script
        self.executable_script_args = executable_script_args

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
            f"tar -xvzf {self.dataset_archive_name} >/dev/null 2>&1",
            "\n",
        ]
        # copy 2019 dataset (training data) aswell if 2021 or InTheWild datasets are used
        if "21" in self.dataset_archive_name or "InTheWild" in self.dataset_archive_name:
            data_script.extend(
                [
                    # copy 2019 dataset
                    f"cp -r $DATADIR/{self.dataset_archive_path}LA19.tar.gz .",  # copy to scratchdir
                    f"tar -xvzf LA19.tar.gz >/dev/null 2>&1",
                    "\n",
                ]
            )

        checkpoint_script = []
        if self.checkpoint_archive_name:
            checkpoint_script = [
                # copy checkpoint
                'echo "Copying checkpoint archive"',
                f"cp $DATADIR/{self.checkpoint_archive_name} .",  # copy to scratchdir
                f'unzip {self.checkpoint_archive_name} "*_20.pt" >/dev/null 2>&1',
                "\n",
            ]

        exec_script = [
            # run the script
            "chmod 755 ./*.py",
            'echo "Running the script"',
            f'./{self.executable_script} {" ".join(self.executable_script_args)} 2>&1',
            "\n",
        ]

        results_script = [
            # copy results
            'echo "Copying results"',
            'find . -type d -name "__pycache__" -exec rm -rf {} +',  # remove __pycache__ directories
            'zip -r "$archivename" classifiers datasets embeddings feature_processors trainers ./*.py ./*.png ./*.pt >/dev/null 2>&1',
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
            "checkpoint_archive_name",
            "executable_script",
            "executable_script_args",
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
    # Modify parameters and arguments here
    for c in ["FFConcat4"]:
        dataset = "InTheWildDataset_pair"
        generate_job_script(
            jobname=f"DP_XLSR_300M_MHFA_{c}_InTheWild_DF21",
            file_name=f"scripts/{c}_InTheWild_DF21.sh",
            project_archive_name="dp.zip",
            dataset_archive_name="InTheWild.tar.gz",
            executable_script="train_and_eval.py",
            executable_script_args=[
                "--metacentrum",
                "--dataset",
                dataset,
                "--extractor",
                "XLSR_300M",
                "--processor",
                "MHFA",
                "--classifier",
                f"{c}",
            ],
        )
