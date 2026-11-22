#!/usr/bin/env python3
import numpy as np

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.curve_speed import CurveSpeed
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource, get_T_FOLLOW
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop, smooth_value
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
J_CRUISE_BP = [0., 10.0, 25., 40.]
A_CRUISE_MIN = -0.5
AGGRESSIVE_ACCEL_BOOST = 0.1  # m/s^2
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

# Lane Change Acceleration: follow closer while signaling left at highway speed, to close in before passing
LANE_CHANGE_ACCEL_MIN_SPEED = 40 * CV.MPH_TO_MS
LANE_CHANGE_ACCEL_T_FOLLOW = [1.1, 0.85, 0.6]  # follow time (s) for mild, medium, strong
LANE_CHANGE_ACCEL_RATE_DOWN = 0.5  # s/s, closing the gap
LANE_CHANGE_ACCEL_RATE_UP = 0.2  # s/s, slowly back to the personality's distance, to avoid hard braking
PARAMS_UPDATE_PERIOD = 1.0  # s


def get_lane_change_accel_t_follow(t_follow_prev, personality, v_ego, left_blinker, enabled, strength, dt):
  t_follow_base = get_T_FOLLOW(personality)
  target = t_follow_base
  if enabled and left_blinker and v_ego > LANE_CHANGE_ACCEL_MIN_SPEED:
    strength = int(np.clip(strength, 0, len(LANE_CHANGE_ACCEL_T_FOLLOW) - 1))
    target = min(t_follow_base, LANE_CHANGE_ACCEL_T_FOLLOW[strength])
  return float(np.clip(target, t_follow_prev - LANE_CHANGE_ACCEL_RATE_DOWN * dt, t_follow_prev + LANE_CHANGE_ACCEL_RATE_UP * dt))


def get_cruise_accel(v_cruise, v_ego, a_cruise_prev, a_target_prev, dt):
  target_accel = smooth_value(v_cruise - v_ego, a_cruise_prev, tau=1.0, dt=dt)
  target_accel = np.clip(target_accel, A_CRUISE_MIN, ACCEL_MAX)
  j_cruise = np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, min(a_cruise_prev, a_target_prev) - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    self.fcw = False
    self.dt = dt
    self.curve_speed = CurveSpeed(dt)

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False

    self.params = Params()
    self.params_frame = 0
    self.lane_change_accel = self.params.get_bool("LaneChangeAcceleration")
    self.lane_change_accel_strength = self.params.get("LaneChangeAccelerationStrength", return_default=True)
    self.t_follow = get_T_FOLLOW()

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def update(self, sm):
    v_ego = sm['carState'].vEgo
    personality = sm['selfdriveState'].personality
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.output_a_target
      self.curve_speed.reset()

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    self.params_frame += 1
    if self.params_frame >= PARAMS_UPDATE_PERIOD / self.dt:
      self.params_frame = 0
      self.lane_change_accel = self.params.get_bool("LaneChangeAcceleration")
      self.lane_change_accel_strength = self.params.get("LaneChangeAccelerationStrength", return_default=True)

    if reset_state:
      self.t_follow = get_T_FOLLOW(personality)
    self.t_follow = get_lane_change_accel_t_follow(self.t_follow, personality, v_ego, sm['carState'].leftBlinker,
                                                   self.lane_change_accel, self.lane_change_accel_strength, self.dt)

    self.mpc.set_weights(prev_accel_constraint, personality=personality)
    if self.CP.openpilotLongitudinalControl:
      self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    else:
      self.mpc.set_cur_state(v_ego, sm['carState'].aEgo)
    self.mpc.update(sm['radarState'], personality=personality, t_follow=self.t_follow)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.output_a_target

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
    curve_offset = self.curve_speed.update(sm['modelV2'], sm['vehicleParameters'].roll)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration + curve_offset
    if personality == log.LongitudinalPersonality.aggressive:
      speed_scale = np.interp(v_ego, [5.0, 10.0], [0.0, 1.0])
      accel_scale = np.interp(output_a_target_e2e, [-1.0, -0.5], [0.0, 1.0])
      output_a_target_e2e += AGGRESSIVE_ACCEL_BOOST * speed_scale * accel_scale
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    self.a_cruise = get_cruise_accel(v_cruise, v_ego, self.a_cruise, self.output_a_target, self.dt)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop),
                  (output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e)]

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates)
    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime - sm.logMonoTime['modelV2']) / 1e9
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True

    pm.send('longitudinalPlan', plan_send)
