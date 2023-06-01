import pickle


def main():
    model = "gnn"
    runs, epochs = 5, 1000
    result = [[] for _ in range(runs)]
    highest = []
    all = [[], []]  # mean, std
    run = 0
    name = "samples/job.sage.proteins.3277.out"
    with open(name, "r") as f:
        f.readline()
        while True:
            line = f.readline()
            line_list = line.split(",")
            if len(line_list) == 1:
                train_highest = float(f.readline().split(":")[-1].strip(" "))
                for _ in range(2):
                    f.readline()
                test_highest = float(f.readline().split(":")[-1].strip(" "))
                highest.append([train_highest / 100, test_highest / 100])
                run += 1

                if run == runs:
                    f.readline()
                    train_all = f.readline().split(":")[-1].strip(" ").strip("\n").split(" ")
                    train_mean, train_std = float(train_all[0]), float(train_all[-1])
                    for _ in range(2):
                        f.readline()
                    test_all = f.readline().split(":")[-1].strip(" ").strip("\n").split(" ")
                    test_mean, test_std = float(test_all[0]), float(test_all[-1])
                    all[0] = [train_mean, test_mean]
                    all[1] = [train_std, test_std]
                    break
                continue
            train_metric, test_metric = line_list[3], line_list[-1]
            train_acc = float(train_metric.split(":")[-1].strip().strip("%"))
            if model == "gnn":
                test_acc = float(test_metric.split(" ")[-1].strip().strip("%"))
            else:
                test_acc = float(test_metric.split(":")[-1].strip().strip("%"))
            result[run - 1].append([train_acc / 100, 0, test_acc / 100, 0, 0, 0, 0])
    mapping = dict()
    mapping["result"] = result
    mapping["baseline"] = [0, 0]
    mapping["highest"] = highest
    mapping["all"] = all
    output_name = name.rstrip(".out") + ".pkl"
    with open(output_name, "wb") as f:
        pickle.dump(mapping, f)


if __name__ == "__main__":
    main()
