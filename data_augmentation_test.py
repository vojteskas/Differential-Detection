import sys
sys.path.append("datasets")
import ASVspoof5 as asv5

if __name__ == "__main__":
    with asv5.ASVspoof5Dataset_single_augmented(algo=4) as single:
        sample = single[0]
        print(sample)
