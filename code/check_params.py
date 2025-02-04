import numpy as np

def main():
    params = np.load('parameters/hdim25_lookback0_lr0.5/U.npy')
    print(params)


if __name__ == "__main__":
    main()