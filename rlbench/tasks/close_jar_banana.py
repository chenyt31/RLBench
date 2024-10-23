from typing import List, Tuple
from rlbench.backend.task import Task
from typing import List
from rlbench.backend.task import Task
from rlbench.const import colors
from rlbench.backend.conditions import NothingGrasped, DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy

class CloseJarBanana(Task):

    def init_task(self) -> None:
        self.lid = Shape('jar_lid0')
        self.jars = [Shape('jar%d' % i) for i in range(2)]
        self.banana = Shape('banana')
        self.register_graspable_objects([self.lid])
        self.boundary = Shape('spawn_boundary')
        self.boundary0 = Shape('spawn_boundary0')
        self.boundary1 = Shape('spawn_boundary1')
        self.conditions = [NothingGrasped(self.robot.gripper)]

    def init_episode(self, index: int) -> List[str]:
        index = index%2
        b0 = SpawnBoundary([self.boundary0])
        b1 = SpawnBoundary([self.boundary1])

        first_jar = self.jars[0]
        b1.sample(first_jar, min_distance=0.01)
        first_jar_position = np.array(first_jar.get_position())

        second_jar = self.jars[1]
        b0.sample(second_jar, min_distance=0.01)
        second_jar_position = np.array(second_jar.get_position())

        if index == 0:
            b1.sample(self.banana, min_distance=0.01)
            banana_position = np.array(self.banana.get_position())
            
            while np.linalg.norm(banana_position - first_jar_position) >= np.linalg.norm(banana_position - second_jar_position):
                b1.sample(self.banana, min_distance=0.01)
                banana_position = np.array(self.banana.get_position())
        else:
            b0.sample(self.banana, min_distance=0.01)
            banana_position = np.array(self.banana.get_position())
            
            while np.linalg.norm(banana_position - second_jar_position) >= np.linalg.norm(banana_position - first_jar_position):
                b0.sample(self.banana, min_distance=0.01)
                banana_position = np.array(self.banana.get_position())


        available_colors = list(range(len(colors)))
        target_color_index = np.random.choice(available_colors)
        available_colors.remove(target_color_index)
        distractor_color_index = np.random.choice(available_colors)

        _, target_color_rgb = colors[target_color_index]
        first_jar.set_color(target_color_rgb)
        _, distractor_color_rgb = colors[distractor_color_index]
        second_jar.set_color(distractor_color_rgb)

        if index == 0:
            success = ProximitySensor('success')
            success.set_position([0.0, 0.0, 0.05], relative_to=first_jar,
                                reset_dynamics=False)

            w3 = Dummy('waypoint3')
            w3.set_orientation([-np.pi, 0, -np.pi], reset_dynamics=False)
            w3.set_position([0.0, 0.0, 0.125], relative_to=first_jar,
                            reset_dynamics=False)
        else:
            success = ProximitySensor('success')
            success.set_position([0.0, 0.0, 0.05], relative_to=second_jar,
                                reset_dynamics=False)

            w3 = Dummy('waypoint3')
            w3.set_orientation([-np.pi, 0, -np.pi], reset_dynamics=False)
            w3.set_position([0.0, 0.0, 0.125], relative_to=second_jar,
                            reset_dynamics=False)

        self.register_success_conditions([DetectedCondition(self.lid, success)])
        # self.conditions += [DetectedCondition(self.lid, success)]
        # self.register_success_conditions(self.conditions)

        return ['close the jar closer to the banana',
                'screw on the jar lid that is closest to the banana',
                'grasping the lid, lift it from the table and use it to seal '
                'the jar closer to the banana',
                'pick up the lid from the table and put it on the jar closest to the banana']

    
    def variation_count(self) -> int:
        return 2

    def cleanup(self) -> None:
        self.conditions = [NothingGrasped(self.robot.gripper)]

    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float],
                                            Tuple[float, float, float]]:
        # This is here to stop the issue of gripper rotation joint reaching its
        # limit and not being able to go through the full range of rotation to
        # unscrew, leading to a weird jitery and tilted cap while unscrewing.
        # Issue occured rarely so is only minor
        return (0.0, 0.0, -0.6*np.pi), (0.0, 0.0, +0.6*np.pi)