import argparse
from rgcn.main import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CGNN')
    parser.add_argument('--dataset', type=str, help="Name of the datasets", required=True)
    parser.add_argument('--num_trials', type=int, help="The number of the repeating experiments", default=50)
    args = parser.parse_args()

    datasets_config = {
        'Cora': {'d_input': 1433,
                 'd_output': 7},
        'Citeseer': {'d_input': 3703,
                     'd_output': 6},
        'PubMed': {'d_input': 500,
                   'd_output': 3},
        'CS': {'d_input': 6805,
               'd_output': 15},
        'Physics': {'d_input': 8415,
                    'd_output': 5},
        'Computers': {'d_input': 767,
                      'd_output': 10},
        'Photo': {'d_input': 745,
                  'd_output': 8},
        'WikiCS': {'d_input': 300,
                   'd_output': 10},
    }

    main(args, datasets_config[args.dataset]['d_input'], datasets_config[args.dataset]['d_output'])