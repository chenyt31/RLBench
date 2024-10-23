from typing import List
import itertools
import math
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import JointCondition, ConditionSet

MAX_TARGET_BUTTONS = 1
MAX_VARIATIONS = 50

# button top plate and wrapper will be be red before task completion
# and be changed to cyan upon success of task, so colors list used to randomly vary colors of
# base block will be redefined, excluding red and green
colors = [
    ('maroon', (0.5, 0.0, 0.0)),
    ('green', (0.0, 0.5, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('navy', (0.0, 0.0, 0.5)),
    ('yellow', (1.0, 1.0, 0.0)),
    ('cyan', (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)),
    ('silver', (0.75, 0.75, 0.75)),
    ('gray', (0.5, 0.5, 0.5)),
    ('orange', (1.0, 0.5, 0.0)),
    ('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    ('teal', (0, 0.5, 0.5)),
    ('azure', (0.0, 0.5, 1.0)),
    ('violet', (0.5, 0.0, 1.0)),
    ('rose', (1.0, 0.0, 0.5)),
    ('black', (0.0, 0.0, 0.0)),
    ('white', (1.0, 1.0, 1.0)),
]

def log_to_file(message):
    with open('/data1/cyt/3d_diffuser_actor/RLBench/debug_log.txt', 'a') as file:
        file.write(message + '\n')



class PushButtonsLight(Task):

    def init_task(self) -> None:
        self.buttons_pushed = 0
        self.color_variation_index = 0
        self.target_buttons = [Shape('push_buttons_target%d' % i)
                               for i in range(3)]
        self.target_topPlates = [Shape('target_button_topPlate%d' % i)
                                 for i in range(3)]
        self.target_joints = [Joint('target_button_joint%d' % i)
                              for i in range(3)]
        self.target_wraps = [Shape('target_button_wrap%d' % i)
                             for i in range(3)]
        self.boundaries = Shape('push_buttons_boundary')
        self.color_bulb = Shape('color_bulb')
        self.light_bulb = Shape('light_bulb')
        # goal_conditions merely state joint conditions for push action for
        # each button regardless of whether the task involves pushing it
        self.goal_conditions = [JointCondition(self.target_joints[n], 0.001)
                                for n in range(3)]

        self.register_waypoint_ability_start(0, self._move_above_next_target)
        self.register_waypoints_should_repeat(self._repeat)

    def init_episode(self, index: int) -> List[str]:
        for tp in self.target_topPlates:
            tp.set_color([1.0, 0.0, 0.0])
        for w in self.target_wraps:
            w.set_color([1.0, 0.0, 0.0])
        # For each color permutation, we want to have 1, 2 or 3 buttons pushed
        color_index = int(index / MAX_TARGET_BUTTONS)
        self.buttons_to_push = 1 + index % MAX_TARGET_BUTTONS
        chosen_color_name, chosen_color_rgb = colors[color_index]
        
        for b in self.target_buttons:
            b.set_color(chosen_color_rgb)
        
        # Set the color of the color bulb (assuming 'color_bulb' is an instance of Shape)
        self.color_bulb.set_color(chosen_color_rgb)

        # Update success conditions to require all buttons to match the chosen color

        self.success_conditions = []
        for i in range(self.buttons_to_push):
            self.success_conditions.append(self.goal_conditions[i])

        self.register_success_conditions(
            [ConditionSet(self.success_conditions, True, False)])
        
        b = SpawnBoundary([self.boundaries])
        for button in self.target_buttons:
            b.sample(button, min_distance=0.1)

        b.sample(self.light_bulb, min_distance=0.1) # color sample



        num_non_targets = 3 - self.buttons_to_push
        
        spare_colors = list(set(colors)
                            - set([colors[color_index]]))
        if len(spare_colors) < num_non_targets:
            raise RuntimeError("Not enough spare colors available to choose from.")

        spare_color_rgbs = []
        for i in range(len(spare_colors)):
            _, rgb = spare_colors[i]
            spare_color_rgbs.append(rgb)

        color_choice_indexes = np.random.choice(range(len(spare_colors)),
                                                size=num_non_targets,
                                                replace=False)
        non_target_index = 0
        for i, button in enumerate(self.target_buttons):
            if i >= self.buttons_to_push:
                _, rgb = spare_colors[color_choice_indexes[non_target_index]]
                button.set_color(rgb)
                non_target_index += 1
        # for i, button in enumerate(self.target_buttons):
        #     if i == 1:
        #         button.set_color([1.0, 1.0, 0.0])
        #     if i == 2:
        #         button.set_color([0.5, 0.0, 0.5])
        return ['push the button with the same color as the light',
                'press the button with the color of the light',
                'press the button with the same color as the light']

    def variation_count(self) -> int:
        return np.minimum(
            len(colors) * MAX_TARGET_BUTTONS, MAX_VARIATIONS)


    def step(self) -> None:
        for i in range(len(self.target_buttons)):
            if self.goal_conditions[i].condition_met() == (True, True):
                self.target_topPlates[i].set_color([0.0, 1.0, 0.0])
                self.target_wraps[i].set_color([0.0, 1.0, 0.0])

    def cleanup(self) -> None:
        self.buttons_pushed = 0

    def _move_above_next_target(self, waypoint):
        if self.buttons_pushed >= self.buttons_to_push:
            print('buttons_pushed:', self.buttons_pushed, 'buttons_to_push:',
                  self.buttons_to_push)
            raise RuntimeError('Should not be here.')
        w0 = Dummy('waypoint0')
        x, y, z = self.target_buttons[self.buttons_pushed].get_position()
        w0.set_position([x, y, z + 0.083])
        w0.set_orientation([math.pi, 0, math.pi])

    def _repeat(self):
        self.buttons_pushed += 1
        return self.buttons_pushed < self.buttons_to_push
