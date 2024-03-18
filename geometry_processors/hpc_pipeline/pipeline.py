from typing import List

from geometry_processors.hpc_pipeline.single_step import HpcSingleStep


class HpcPipeline:
    def __init__(self, single_steps: List[HpcSingleStep]) -> None:
        self.single_steps = single_steps

    def run(self):
        prev_id = None
        for single_step in self.single_steps:
            prev_id = single_step.run(prev_id)
        print("Success")