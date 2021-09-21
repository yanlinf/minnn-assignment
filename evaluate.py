import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('f1')
    parser.add_argument('f2')
    args = parser.parse_args()

    with open(args.f1, 'r', encoding='utf-8') as f1, open(args.f2, 'r', encoding='utf-8') as f2:
        n_correct, n_total = 0, 0
        for l1, l2 in zip(f1, f2):
            n_total += 1
            if l1[0] == l2[0]:
                n_correct += 1
    print(f'Accuracy: {n_correct / n_total}')


if __name__ == '__main__':
    main()
