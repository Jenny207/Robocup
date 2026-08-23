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
            1: (-26, 0, 0), #守门员
            2: (-10, 0, 0),
            3: (-3, 10, 0),
            4: (-2, 6, 0),
            5: (-3, -10, 0),
            6: (-2, -6, 0),
            7: (-6, 0, 0),
        },
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
    # 状态机：根据比赛状态和当前行为，决定下一步的行为
    def update_current_behavior(self) -> None:
        """
        Chooses what the agent should do in the current step.

        This function checks the game state and decides which behavior
        or skill should be executed next.
        """
        # 状态1：比赛结束
        if self.agent.world.playmode is PlayModeEnum.GAME_OVER:
            return
        # 状态2：Beam状态（开局站位，进球后复位）
        if self.agent.world.playmode_group in (
            PlayModeGroupEnum.ACTIVE_BEAM,
            PlayModeGroupEnum.PASSIVE_BEAM,
        ):
            self.agent.server.commit_beam(
                pos2d=self.BEAM_POSES[type(self.agent.world.field)][self.agent.world.number][:2],
                rotation=self.BEAM_POSES[type(self.agent.world.field)][self.agent.world.number][2],
            )
            return
        #摔倒后站起
        if self.is_getting_up or self.agent.skills_manager.is_ready(skill_name="GetUp"):
            self.is_getting_up = not self.agent.skills_manager.execute(skill_name="GetUp")

        # 赛前或进球后，原地待命，等待服务器分配站位 (Beam)
        elif self.agent.world.playmode in (
            PlayModeEnum.BEFORE_KICK_OFF,
            PlayModeEnum.OUR_GOAL,
            PlayModeEnum.THEIR_GOAL,
        ):
            self.agent.skills_manager.execute("Neutral")

        elif self.agent.world.playmode is PlayModeEnum.PLAY_ON:
            if self.agent.world.number == 1:
                self.goalkeeper()
            else:
                self.carry_ball()

        # 我方发球系列（开球、角球、任意球等）
        elif self.agent.world.playmode_group == PlayModeGroupEnum.OUR_KICK:

            if self.agent.world.playmode is PlayModeEnum.OUR_KICK_OFF:
                self.execute_kick_off()

        self.agent.robot.commit_motor_targets_pd()

    def goalkeeper(self):
        """
        守门员行为：守在己方球门前方，根据球的位置移动并调整朝向以防守射门。
        """
        # 获取己方球门位置、球位置和自身位置（仅取 x, y 二维坐标）
        our_goal_pos = self.agent.world.field.get_our_goal_position()[:2]
        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]

        field_width = self.agent.world.field.get_width()
        field_length = self.agent.world.field.get_length()

        # 守门员活动区域参数：距球门线 1.5m，Y 轴覆盖范围 ±6m（共 12m）
        goal_area_depth = 1.5
        goal_area_width = 12.0

        # 球门线 X 坐标及守门员 Y 轴活动范围
        goal_left_x = our_goal_pos[0]
        goal_left_y_min = -goal_area_width / 2
        goal_left_y_max = goal_area_width / 2

        # 目标位置：X 轴固定在球门前方 goal_area_depth 处，
        # Y 轴跟随球的横向位置移动，并限制在球门宽度范围内
        target_x = goal_left_x + goal_area_depth
        target_y = ball_pos[1]

        target_y = np.clip(target_y, goal_left_y_min, goal_left_y_max)

        # 计算守门员与球的距离，用于决定朝向策略
        ball_dist = np.linalg.norm(ball_pos - my_pos)

        # 朝向策略：根据球与守门员的距离选择不同的朝向逻辑
        if ball_dist < 1.0:
            # 近身威胁（球距 < 1.0m）：球已逼近球门，需要封堵射门角度
            # 计算"球→球门"向量，守门员面向该方向以挡住射门路线
            ball_to_goal = our_goal_pos - ball_pos
            bg_norm = np.linalg.norm(ball_to_goal)
            if bg_norm > 0:
                # 正常情况：将向量转换为角度作为目标朝向
                desired_orientation = MathOps.vector_angle(ball_to_goal)
            else:
                # 边界情况：球恰好在球门位置，朝向设为 0（面向 +X 方向）
                desired_orientation = 0
        else:
            # 远距离（球距 >= 1.0m）：球还未构成直接威胁
            # 守门员面向球的方向，持续跟踪球的移动，为后续防守做准备
            desired_orientation = MathOps.vector_angle(ball_pos - my_pos)

        # 执行 Walk 技能：移动到目标位置（绝对坐标），并调整朝向
        self.agent.skills_manager.execute(
            "Walk",
            target_2d=(target_x, target_y),
            is_target_absolute=True,
            orientation=desired_orientation
        )

     

    def execute_kick_off(self):
        """
        我方开球行为：
        - 守门员(1号)继续防守球门；
        - 指定一名距球最近的非守门员球员（由初始站位决定）作为开球手，
          靠近球并朝对方球门方向踢球；
        - 其余非守门员球员移动到各自初始站位附近散开，准备接应，
          同时保持在中圈外（不抢球）。
        """
        my_number = self.agent.world.number

        # 守门员不参与开球，继续执行防守逻辑
        if my_number == 1:
            self.goalkeeper()
            return

        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]
        their_goal_pos = self.agent.world.field.get_their_goal_position()[:2]

        # 根据初始站位确定开球手：除守门员外距球（原点）最近者
        beam_poses = self.BEAM_POSES[type(self.agent.world.field)]
        kicker_number = min(
            (n for n in beam_poses if n != 1),
            key=lambda n: np.linalg.norm(np.array(beam_poses[n][:2]))
        )

        if my_number == kicker_number:
            # 开球手：靠近球并朝对方球门方向踢球（复用带球进攻逻辑）
            self.carry_ball()
        else:
            # 其余球员：回到各自初始站位附近散开，面向对方球门准备接应
            target_2d = beam_poses[my_number][:2]
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=target_2d,
                is_target_absolute=True,
                orientation=MathOps.vector_angle(their_goal_pos - my_pos),
            )

    def kick_ball(self):
        """
        开球手踢球行为（无专用踢球技能时，靠行走动量将球踢出）：
        - 先移动到球后方并对准踢球方向；
        - 对齐就位后，沿踢球方向短距离前冲，靠行走动量将球踢出进入比赛。
        """
        their_goal_pos = self.agent.world.field.get_their_goal_position()[:2]
        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]

        # 球→对方球门方向（即踢球方向）
        ball_to_goal = their_goal_pos - ball_pos
        bg_norm = np.linalg.norm(ball_to_goal)
        if bg_norm == 0:
            return
        kick_dir = ball_to_goal / bg_norm

        # 球后方站位点：在踢球方向反侧退后一段距离，确保踢球时朝向球门
        stand_pos = ball_pos - kick_dir * 0.30

        # 自身到球的方向与距离
        my_to_ball = ball_pos - my_pos
        my_to_ball_norm = np.linalg.norm(my_to_ball)
        my_to_ball_dir = my_to_ball / my_to_ball_norm if my_to_ball_norm > 1e-6 else np.zeros(2)

        # 对齐判定：自身→球 方向与踢球方向夹角足够小
        cosang = np.clip(np.dot(my_to_ball_dir, kick_dir), -1.0, 1.0)
        angle_diff = np.arccos(cosang)
        ANGLE_TOL = np.deg2rad(10.0)

        # 是否已在球后方（自身在踢球方向上位于球的后侧）
        behind_ball = np.dot(my_pos - ball_pos, kick_dir) < 0

        desired_orientation = MathOps.vector_angle(kick_dir)

        if not behind_ball or angle_diff > ANGLE_TOL:
            # 未就位：先走到球后方站位点并对准球门，准备踢球
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=stand_pos,
                is_target_absolute=True,
                orientation=desired_orientation,
            )
        else:
            # 已就位：沿踢球方向短距离前冲，靠行走动量将球踢出进入比赛
            # （开球只需把球触动/踢出即可，无需一路带到对方球门）
            #在球的位置沿踢球方向（球→对方球门方向单位向量 kick_dir ）向前延伸 1.5m，
            # 得到一个"球前方短距离点"作为行走终点。
            kick_target = ball_pos + kick_dir * 1.5
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=kick_target,
                is_target_absolute=True,
                orientation=desired_orientation,
            )

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

