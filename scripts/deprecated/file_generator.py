"""
Private code used to generate large quantities of files
"""
import os
import os.path as osp


def gen_config():
    file_format = "../../tmp/configs/config-exp322-rand{}.txt"
    for i in range(50):
        out_file = file_format.format(i)
        os.makedirs(osp.dirname(out_file), exist_ok=True)

        with open("../config.txt") as f:
            content = f.read()

        with open(out_file, "w") as out_f:
            out_f.write(content.format(i, i, i))


def gen_sub_job():
    file_out = "../../tmp/subjob-exp322-rand-1.pbs"
    with open("../utils/file_templates/subjob-amd-head.pbs") as f:
        head_txt = f.read()
    with open("../utils/file_templates/subjob-amd-body.pbs") as f:
        body_txt = f.read()
    with open(file_out, "w") as f:
        f.write(head_txt)
        for i in range(50):
            f.write(body_txt.format(i))


if __name__ == '__main__':
    gen_config()
    gen_sub_job()
