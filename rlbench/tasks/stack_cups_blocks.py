from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.backend.spawn_boundary import SpawnBoundary

DISTRACTORS = 4
class StackCupsBlocks(Task):

    def init_task(self) -> None:
        success_sensor = ProximitySensor('success')
        self.cup1 = Shape('cup1')
        self.cup2 = Shape('cup2')
        self.cup3 = Shape('cup3')
        self.cup1_visual = Shape('cup1_visual')
        self.cup2_visual = Shape('cup2_visual')
        self.cup3_visaul = Shape('cup3_visual')

        self.boundary = SpawnBoundary([Shape('boundary')])

        self.register_graspable_objects([self.cup1, self.cup2, self.cup3])
        self.register_success_conditions([
            DetectedCondition(self.cup1, success_sensor),
            DetectedCondition(self.cup3, success_sensor),
            NothingGrasped(self.robot.gripper)
        ])

        self.distractors = [
            Shape('stack_blocks_distractor%d' % i)
            for i in range(DISTRACTORS)]
        self.boundaries = [Shape('stack_blocks_boundary%d' % i)
                           for i in range(4)]

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index
        target_color_name, target_rgb = colors[index]

        random_idx = np.random.choice(len(colors))
        while random_idx == index:
            random_idx = np.random.choice(len(colors))
        _, other1_rgb = colors[random_idx]

        random_idx = np.random.choice(len(colors))
        while random_idx == index:
            random_idx = np.random.choice(len(colors))
        _, other2_rgb = colors[random_idx]

        self.cup2_visual.set_color(target_rgb)
        self.cup1_visual.set_color(other1_rgb)
        self.cup3_visaul.set_color(other2_rgb)

        self.boundary.clear()
        self.boundary.sample(self.cup2, min_distance=0.05,
                             min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        self.boundary.sample(self.cup1, min_distance=0.05,
                             min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        self.boundary.sample(self.cup3, min_distance=0.05,
                             min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        
        for i, block in enumerate(self.distractors):
            if i < DISTRACTORS-1:
                block.set_color(target_rgb)
            else:
                block.set_color(other2_rgb)


        # bs = SpawnBoundary(self.boundaries)
        for block in self.distractors:
            self.boundary.sample(block, min_distance=0.1)

        return ['identify the color with the most blocks, choose that color for the base cup, and stack the other cups on top of it.']

    def variation_count(self) -> int:
        return len(colors)
