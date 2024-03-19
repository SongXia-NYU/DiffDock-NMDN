from typing import List

from geometry_processors.lazy_property import lazy_property


class HPC_FilePurgeVis:
    def __init__(self, file_list_txt: str, ignored_folders: List[str]) -> None:
        self.file_list_txt = file_list_txt
        self.ignored_folders = set(ignored_folders)

    def vis_level(self, level: int):
        level_split = ["/".join(split[:level]) for split in self.splits]
        level_split = list(set(level_split))
        level_split.sort()
        for folder in level_split:
            print(folder)

    @lazy_property
    def depth(self) -> int:
        return max([len(split) for split in self.splits])

    @lazy_property
    def splits(self) -> List[List[str]]:
        with open(self.file_list_txt) as f:
            lines_unfiltered = [line for line in f.readlines()]
        splits = []
        for line in lines_unfiltered:
            ignore: bool = False
            for ignored_folder in self.ignored_folders:
                if line.startswith(ignored_folder):
                    ignore = True
                    break
            if ignore:
                continue
            splits.append(line.split("/"))
        return splits

if __name__ == "__main__":
    ignored = []
    vis = HPC_FilePurgeVis("/scratch/cleanup/60days-files/20231130/sx801", ignored)
    vis.vis_level(4)
