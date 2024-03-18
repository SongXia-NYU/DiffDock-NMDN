if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--amd", action="store_true")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--al", action="store_true")
    args = parser.parse_args()
    f_path = "../utils/file_templates/subjob{}{}{}.pbs".\
        format("-al" if args.al else "",
               "-amd" if args.amd else "",
               "-dist" if args.dist else "")

    start = args.start
    end = args.end+1 if args.end > 0 else start + 1
    for i in range(start, end):
        with open(f_path) as f_template:
            with open("../subjob-exp{}.pbs".format(i), "w") as f_out:
                f_out.write(f_template.read().format(i, i))
