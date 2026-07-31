from dataclasses import Field
import logging
from typing import Mapping

import numpy as np
from mujococodebase.utils.math_ops import MathOps
from mujococodebase.world.field import CustomField, FIFAField, HLAdultField
from mujococodebase.world.play_mode import PlayModeEnum, PlayModeGroupEnum


logger = logging.getLogger()


class DecisionMaker:
    """
    Responsible for deciding what the agent should do at each moment.

    This class is called every simulation step to update the agent's behavior
    based on the current state of the world and game conditions.
    """

    BEAM_POSES: Mapping[type[Field], Mapping[int, tuple[float, float, float]]] ={
        FIFAField: {
            1: (2.1, 0, 0),
            2: (22.0, 12.0, 0),
            3: (22.0, 4.0, 0),
            4: (22.0, -4.0, 0),
            5: (22.0, -12.0, 0),
            6: (15.0, 0.0, 0),
            7: (4.0, 16.0, 0),
            8: (11.0, 6.0, 0),
            9: (11.0, -6.0, 0),
            10: (4.0, -16.0, 0),
            11: (7.0, 0.0, 0),
        },
        HLAdultField: {
            1: (7.0, 0.0, 0),
            2: (2.0, -1.5, 0),
            3: (2.0, 1.5, 0),
        },
        CustomField: {
            1: (-25.0, 0.0, 0),
            2: (-15.0, -12.0, 0),
            3: (-15.0, 12.0, 0),
            4: (0.0, -10.0, 0),
            5: (0.0, 10.0, 0),
            6: (15.0, -8.0, 0),
            7: (15.0, 8.0, 0),
        }
    } 

    def __init__(self, agent):
        """
        Creates a new DecisionMaker linked to the given agent.

        Args:
            agent: The main agent that owns this DecisionMaker.
        """
        from mujococodebase.agent import Agent  # type hinting

        self.agent: Agent = agent
        self.is_getting_up: bool = False

    def update_current_behavior(self) -> None:
        """
        Chooses what the agent should do in the current step.

        This function checks the game state and decides which behavior
        or skill should be executed next.
        """

        if self.agent.world.playmode is PlayModeEnum.GAME_OVER:
            return

        if self.agent.world.playmode_group in (
            PlayModeGroupEnum.ACTIVE_BEAM,
            PlayModeGroupEnum.PASSIVE_BEAM,
        ):
            self.agent.server.commit_beam(
                pos2d=self.BEAM_POSES[type(self.agent.world.field)][self.agent.world.number][:2],
                rotation=self.BEAM_POSES[type(self.agent.world.field)][self.agent.world.number][2],
            )

        if self.is_getting_up or self.agent.skills_manager.is_ready(skill_name="GetUp"):
            self.is_getting_up = not self.agent.skills_manager.execute(skill_name="GetUp")

        elif self.agent.world.playmode is PlayModeEnum.PLAY_ON:
            if self.agent.world.number == 1:
                self.goalkeeper()
            else:
                self.carry_ball()

        elif self.agent.world.playmode in (
            PlayModeEnum.BEFORE_KICK_OFF,
            PlayModeEnum.OUR_GOAL,
            PlayModeEnum.THEIR_GOAL,
        ):
            self.agent.skills_manager.execute("Neutral")
        #敌方开球时的防守行为 ，避免犯规
        elif self.agent.world.playmode_group is PlayModeGroupEnum.THEIR_KICK:
            self.defend_kick()
        #我方开球时执行进攻
        elif self.agent.world.playmode_group is PlayModeGroupEnum.OUR_KICK:
            self.execute_our_kick()

        else:
            if self.agent.world.number == 1:
                self.goalkeeper()
            else:
                self.carry_ball()

        self.agent.robot.commit_motor_targets_pd()

    def goalkeeper(self):
        """
        Goalkeeper behavior: stays near our goal and defends against shots.
        """
        our_goal_pos = self.agent.world.field.get_our_goal_position()[:2]
        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]

        field_width = self.agent.world.field.get_width()
        field_length = self.agent.world.field.get_length()

        goal_area_depth = 5.0
        goal_area_width = 12.0

        goal_left_x = our_goal_pos[0]
        goal_left_y_min = -goal_area_width / 2
        goal_left_y_max = goal_area_width / 2

        target_x = goal_left_x + goal_area_depth
        target_y = ball_pos[1]

        target_y = np.clip(target_y, goal_left_y_min, goal_left_y_max)

        ball_dist = np.linalg.norm(ball_pos - my_pos)

        if ball_dist < 1.0:
            ball_to_goal = our_goal_pos - ball_pos
            bg_norm = np.linalg.norm(ball_to_goal)
            if bg_norm > 0:
                desired_orientation = MathOps.vector_angle(ball_to_goal)
            else:
                desired_orientation = 0
        else:
            desired_orientation = MathOps.vector_angle(ball_pos - my_pos)

        self.agent.skills_manager.execute(
            "Walk",
            target_2d=(target_x, target_y),
            is_target_absolute=True,
            orientation=desired_orientation
        )

    def defend_kick(self):
        """
        Defensive behavior when the opponent has a set play.
        Players maintain a safe distance from the ball to avoid fouls.
        """
        if self.agent.world.number == 1:
            self.goalkeeper()
            return

        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]

        MIN_BALL_DISTANCE = 1.0

        ball_to_me = my_pos - ball_pos
        dist_to_ball = np.linalg.norm(ball_to_me)

        if dist_to_ball < MIN_BALL_DISTANCE:
            if dist_to_ball > 1e-6:
                retreat_dir = ball_to_me / dist_to_ball
            else:
                retreat_dir = np.array([1.0, 0.0])

            target_pos = ball_pos + retreat_dir * MIN_BALL_DISTANCE
            desired_orientation = MathOps.vector_angle(ball_pos - my_pos)

            self.agent.skills_manager.execute(
                "Walk",
                target_2d=target_pos,
                is_target_absolute=True,
                orientation=desired_orientation
            )
        else:
            desired_orientation = MathOps.vector_angle(ball_pos - my_pos)
            self.agent.skills_manager.execute("Neutral")

    def execute_our_kick(self):
        """
        Behavior when our team has a set play.
        Goalkeeper defends, field players approach the ball to restart play.
        """
        if self.agent.world.number == 1:
            self.goalkeeper()
        else:
            self.carry_ball()

    def carry_ball(self):
        """
        Basic example of a behavior: moves the robot toward the goal while handling the ball.
        """
        their_goal_pos = self.agent.world.field.get_their_goal_position()[:2]
        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]

        ball_to_goal = their_goal_pos - ball_pos
        bg_norm = np.linalg.norm(ball_to_goal)
        if bg_norm == 0:
            return 
        ball_to_goal_dir = ball_to_goal / bg_norm

        dist_from_ball_to_start_carrying = 0.30
        carry_ball_pos = ball_pos - ball_to_goal_dir * dist_from_ball_to_start_carrying

        my_to_ball = ball_pos - my_pos
        my_to_ball_norm = np.linalg.norm(my_to_ball)
        if my_to_ball_norm == 0:
            my_to_ball_dir = np.zeros(2)
        else:
            my_to_ball_dir = my_to_ball / my_to_ball_norm

        cosang = np.dot(my_to_ball_dir, ball_to_goal_dir)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle_diff = np.arccos(cosang)

        ANGLE_TOL = np.deg2rad(7.5)
        aligned = (my_to_ball_norm > 1e-6) and (angle_diff <= ANGLE_TOL)

        behind_ball = np.dot(my_pos - ball_pos, ball_to_goal_dir) < 0
        desired_orientation = MathOps.vector_angle(ball_to_goal)

        if not aligned or not behind_ball:
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=carry_ball_pos,
                is_target_absolute=True,
                orientation=None if np.linalg.norm(my_pos - carry_ball_pos) > 2 else desired_orientation
            )
        else:
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=their_goal_pos,
                is_target_absolute=True,
                orientation=desired_orientation
            )

