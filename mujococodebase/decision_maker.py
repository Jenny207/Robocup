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
             self.handle_play_on()

        # 我方发球系列（开球、角球、任意球等）
        elif self.agent.world.playmode_group == PlayModeGroupEnum.OUR_KICK:

            if self.agent.world.playmode is PlayModeEnum.OUR_KICK_OFF:
                self.execute_kick_off()

        self.agent.robot.commit_motor_targets_pd()

    def handle_play_on(self) -> None:
        """
        比赛正常进行时的逻辑：引入简单的动态角色分配
        """
        w = self.agent.world
        my_number = w.number
        ball_pos = w.ball_pos[:2]

        # 求出离球最近的机器人号码
        active_play_unum = self.get_closest_teammate_to_ball()

        # 守门员始终执行守门逻辑
        if my_number == 1:
            self.goalkeeper()
            return

        # 活跃球员（离球最近的球员）执行带球进攻
        if my_number == active_play_unum:
            self.carry_ball_optimized()
            return

        # 非活跃球员根据球的位置决定进攻或防守站位
        if ball_pos[0] > 0:
            # 球在对方半场，执行进攻站位
            self.offensive_logic()
        else:
            # 球在我方半场，执行防守站位
            self.defender_logic()

    def get_closest_teammate_to_ball(self) -> int:
        """
        求出离球最近的队友号码（排除守门员1号）。
        - 自身位置用 world.global_position（实时可靠，因自身不感知自己，
          our_team_players 中自身条目为零向量且永未被观测，不可用）；
        - 其余队友用 world.our_team_players，仅当 last_seen_time 不为 None 且
          观测时间在 server_time 的 0.5s 内时位置才可信（过滤从未观测与陈旧观测，
          避免零向量被误判为"在球场中心"或用过期位置算距离）；
        - 守门员1号不参与争抢，排除，避免出现"无人带球"的空档。
        若无任何有效候选（队友均未被观测或已过期），回退为7号作为兜底，
        使各 agent 在无数据时对"谁是活跃球员"达成一致，避免全员自认活跃导致撞车。
        """
        w = self.agent.world
        ball_pos = w.ball_pos[:2]
        num_players = w.MAX_PLAYERS_PER_TEAM

        min_dist = float('inf')
        closest_unum = 7  # 兜底：无有效候选时全队一致指向7号

        for p_id in range(2, num_players + 1):
            dist = float('inf')
            if p_id == w.number:
                # 自身位置实时可用，始终作为候选
                dist = np.linalg.norm(w.global_position[:2] - ball_pos)
            else:
                teammate = w.our_team_players[p_id - 1]
                # 仅近期(<=0.5s)被观测过的队友位置才可信
                if (teammate.last_seen_time is not None
                        and w.server_time is not None
                        and w.server_time - teammate.last_seen_time < 0.5):
                    dist = np.linalg.norm(teammate.position[:2] - ball_pos)

            if dist < min_dist:
                min_dist = dist
                closest_unum = p_id

        return closest_unum

    def carry_ball_optimized(self):
        """
        带球进攻（优化版）：相比 carry_ball，
        - 远距离且未就位时直接奔向球（终点=球），加快接近、减少绕行；
        - 近距离切换到球后方就位点对齐；
        - 已对齐且在球后方时，沿球门方向带球推进。
        """
        their_goal_pos = self.agent.world.field.get_their_goal_position()[:2]
        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]

        # 球→对方球门方向及其单位向量
        ball_to_goal = their_goal_pos - ball_pos
        bg_norm = np.linalg.norm(ball_to_goal)
        if bg_norm == 0:
            return
        ball_to_goal_dir = ball_to_goal / bg_norm

        # 球后方就位点
        carry_ball_pos = ball_pos - ball_to_goal_dir * 0.30

        # 自身→球 方向与距离，用于对齐判定
        my_to_ball = ball_pos - my_pos
        my_to_ball_norm = np.linalg.norm(my_to_ball)
        my_to_ball_dir = my_to_ball / my_to_ball_norm if my_to_ball_norm > 1e-6 else np.zeros(2)

        # 夹角与对齐、球后方判定
        cosang = np.clip(np.dot(my_to_ball_dir, ball_to_goal_dir), -1.0, 1.0)
        angle_diff = np.arccos(cosang)
        ANGLE_TOL = np.deg2rad(7.5)
        aligned = (my_to_ball_norm > 1e-6) and (angle_diff <= ANGLE_TOL)
        behind_ball = np.dot(my_pos - ball_pos, ball_to_goal_dir) < 0
        desired_orientation = MathOps.vector_angle(ball_to_goal)

        if my_to_ball_norm > 1.0 and (not aligned or not behind_ball):
            # 远距离未就位：直接奔向球，优先赶路（不强制朝向）
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=ball_pos,
                is_target_absolute=True,
                orientation=None,
            )
        elif not aligned or not behind_ball:
            # 近距离未就位：走球后方就位点并转向球门
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=carry_ball_pos,
                is_target_absolute=True,
                orientation=desired_orientation,
            )
        else:
            # 已就位对齐：沿球门方向带球推进
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=their_goal_pos,
                is_target_absolute=True,
                orientation=desired_orientation,
            )

    def offensive_logic(self):
        """
        非活跃球员的进攻站位：球在对方半场时，
        球员散开到球后方（介于中场与球之间）提供接应点，
        横向按各自初始站位 Y 保持间距，朝向对方球门。
        """
        w = self.agent.world
        ball_pos = w.ball_pos[:2]
        my_pos = w.global_position[:2]
        their_goal_pos = w.field.get_their_goal_position()[:2]
        # 横向间距用初始站位 Y，避免与队友重叠
        my_y = self.BEAM_POSES[type(w.field)][w.number][1]

        # 站在球后方 3m，但不退过中场线（保持压上进攻）
        target_x = max(ball_pos[0] - 3.0, 0.0)
        target_2d = (target_x, my_y)

        self.agent.skills_manager.execute(
            "Walk",
            target_2d=target_2d,
            is_target_absolute=True,
            orientation=MathOps.vector_angle(their_goal_pos - my_pos),
        )

    def defender_logic(self):
        """
        非活跃球员的防守站位：球在我方半场时，
        球员回撤到球与己方球门之间，封堵进攻路线/覆盖空间，
        横向按各自初始站位 Y 保持间距，朝向球。
        """
        w = self.agent.world
        ball_pos = w.ball_pos[:2]
        my_pos = w.global_position[:2]
        our_goal_pos = w.field.get_our_goal_position()[:2]
        # 横向间距用初始站位 Y
        my_y = self.BEAM_POSES[type(w.field)][w.number][1]

        # 站在球与己方球门的中点（位于球门前、拦截路线上）
        target_x = (ball_pos[0] + our_goal_pos[0]) / 2.0
        target_2d = (target_x, my_y)

        self.agent.skills_manager.execute(
            "Walk",
            target_2d=target_2d,
            is_target_absolute=True,
            orientation=MathOps.vector_angle(ball_pos - my_pos),
        )

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
            self.kick_ball()
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
        带球进攻行为：在保持控球的同时把球向对方球门方向推进。
        策略——始终绕到"球→对方球门"连线的后方就位，对齐后沿该方向带球前进。
        """
        # 取二维坐标（x, y）：对方球门、球、自身位置
        their_goal_pos = self.agent.world.field.get_their_goal_position()[:2]
        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]

        # 球指向对方球门的向量（有长度、有方向）
        ball_to_goal = their_goal_pos - ball_pos
        # 该向量的模长（球到球门的直线距离）
        bg_norm = np.linalg.norm(ball_to_goal)
        if bg_norm == 0:
            # 边界情况：球恰好在球门位置，无需带球推进
            return
        # 向量 ÷ 自身模长 = 同方向的单位向量（带球推进方向）
        ball_to_goal_dir = ball_to_goal / bg_norm

        # 带球就位点：在球后方 0.30m 处（推进方向反侧）
        # 站在这里面朝球门向前走，脚能持续推动球前进而不易丢球
        dist_from_ball_to_start_carrying = 0.30
        carry_ball_pos = ball_pos - ball_to_goal_dir * dist_from_ball_to_start_carrying

        # 自身→球 向量及其模长，用于判断自身相对球的朝向是否对齐
        my_to_ball = ball_pos - my_pos
        my_to_ball_norm = np.linalg.norm(my_to_ball)
        if my_to_ball_norm == 0:
            # 边界情况：自身恰好在球上，方向无法归一化，置零向量避免除零
            my_to_ball_dir = np.zeros(2)
        else:
            my_to_ball_dir = my_to_ball / my_to_ball_norm

        # 用点积求"自身→球"方向与"带球推进"方向的夹角余弦
        # 两单位向量点积 = cosθ，θ 为两方向夹角
        cosang = np.dot(my_to_ball_dir, ball_to_goal_dir)
        # 裁剪到 [-1,1] 防止浮点误差使 arccos 越界报错
        cosang = np.clip(cosang, -1.0, 1.0)
        angle_diff = np.arccos(cosang)

        # 对齐判定：夹角 <= 7.5° 且确实存在到球的距离（避免与球重合时误判对齐）
        ANGLE_TOL = np.deg2rad(7.5)
        aligned = (my_to_ball_norm > 1e-6) and (angle_diff <= ANGLE_TOL)

        # 是否已在球后方：把"球→自身"向量投影到推进方向上，
        # 投影为负说明自身位于推进方向的反侧（即球的后面），可以直接向前带球
        behind_ball = np.dot(my_pos - ball_pos, ball_to_goal_dir) < 0
        # 期望朝向：始终面朝对方球门方向
        desired_orientation = MathOps.vector_angle(ball_to_goal)

        if not aligned or not behind_ball:
            # 未就位：绕到球后方就位点 carry_ball_pos
            # 距就位点 > 2m 时先不强制朝向（orientation=None，优先赶路）
            # 进入 2m 内再开始转向球门，便于最后对齐
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=carry_ball_pos,
                is_target_absolute=True,
                orientation=None if np.linalg.norm(my_pos - carry_ball_pos) > 2 else desired_orientation
            )
        else:
            # 已对齐且在球后方：沿球门方向带球前进，目标点直设为对方球门
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=their_goal_pos,
                is_target_absolute=True,
                orientation=desired_orientation
            )

