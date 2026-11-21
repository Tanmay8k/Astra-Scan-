import numpy as np
from ai_trainer import extract_features_from_dem, load_or_train_model
from database_vault import SouthPoleVault


class AutonomousSafetyEngine:
    def __init__(self, vault: SouthPoleVault):
        self.vault = vault
        bundle = load_or_train_model()
        self.model = bundle["model"]
        self.model_accuracy = bundle["accuracy"]

        # Comms & Safety Backup State
        self.mode = "AUTONOMOUS_EXPLORE"
        self.signal_loss_ticks = 0
        self.recovery_wait_limit = 2  # Ticks to wait for signal recovery before SAFETY_BACKUP
        self.retrace_queue: list[tuple[int, int]] = []

    def predict_forward_corridor(
        self,
        dem: np.ndarray,
        thermal_k: np.ndarray,
        curr_pos: tuple[int, int],
        heading: str,
        lookahead_steps: int = 6,
    ) -> dict:
        """
        ADAS / Self-Driving Style Forward Lookahead:
        Predicts terrain safety across Left, Center, and Right lanes 10m to 60m ahead
        BEFORE the rover enters the zone.
        """
        rows, cols = dem.shape
        r, c = curr_pos
        dir_vec = {
            "NORTH": (-1, 0),
            "SOUTH": (1, 0),
            "EAST": (0, 1),
            "WEST": (0, -1),
            "SOUTH-EAST": (1, 1),
        }.get(heading, (-1, 0))

        # Perpendicular vector for Left (-1) and Right (+1) corridor lanes
        perp_vec = (-dir_vec[1], dir_vec[0])

        gy, gx = np.gradient(dem * 16.0)
        slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
        roughness = np.abs(gx) + np.abs(gy)

        lanes = {"LEFT_LANE": -1, "CENTER_LANE": 0, "RIGHT_LANE": 1}
        corridor_report = {}

        for lane_name, offset in lanes.items():
            lane_preds = []
            hazard_distance_m = None
            for step in range(1, lookahead_steps + 1):
                pr = int(np.clip(r + dir_vec[0] * step + perp_vec[0] * offset, 0, rows - 1))
                pc = int((c + dir_vec[1] * step + perp_vec[1] * offset) % cols)

                feat = np.array([[dem[pr, pc], slope_deg[pr, pc], thermal_k[pr, pc], roughness[pr, pc]]])
                pred_lbl = str(self.model.predict(feat)[0])
                lane_preds.append({"step_m": step * 10, "pos": (pr, pc), "prediction": pred_lbl})

                if pred_lbl == "CRATER_WALL_HAZARD" and hazard_distance_m is None:
                    hazard_distance_m = step * 10

            corridor_report[lane_name] = {
                "clear": hazard_distance_m is None,
                "hazard_in_m": hazard_distance_m,
                "waypoints": lane_preds,
            }

        # Decide autonomous steering recommendation
        if corridor_report["CENTER_LANE"]["clear"]:
            action_advice = "PROCEED_CENTER_NOMINAL"
        elif corridor_report["LEFT_LANE"]["clear"]:
            action_advice = "STEER_LEFT_AVOID_HAZARD"
        elif corridor_report["RIGHT_LANE"]["clear"]:
            action_advice = "STEER_RIGHT_AVOID_HAZARD"
        else:
            action_advice = "ENGAGE_THRUSTER_JUMP_OR_HALT"

        return {
            "lookahead_range_m": lookahead_steps * 10,
            "recommended_action": action_advice,
            "corridors": corridor_report,
        }

    def evaluate_comms_and_backup(
        self, curr_pos: tuple[int, int], signal_pct: float
    ) -> tuple[tuple[int, int], str]:
        """
        Signal Watchdog & 10-Minute SAFETY_BACKUP Retrace Controller:
        1. If signal >= 25%: Normal operations + flush any buffered South Pole data.
        2. If signal < 25%: Attempts recovery for `recovery_wait_limit` ticks.
        3. If still lost: Engages SAFETY_BACKUP and retraces the last 10 mins of breadcrumbs.
        """
        if signal_pct >= 25.0:
            was_in_backup = self.mode in ("SAFETY_BACKUP", "SIGNAL_RECOVERY_WAIT")
            self.signal_loss_ticks = 0
            self.retrace_queue.clear()
            self.mode = "AUTONOMOUS_EXPLORE"
            if was_in_backup:
                flushed = self.vault.flush_downlink_queue()
                return curr_pos, f"SIGNAL_RESTORED_FLUSHED_{flushed}_IMAGES"
            return curr_pos, "COMMS_LINK_NOMINAL"

        # Signal lost (< 25%)
        self.signal_loss_ticks += 1
        if self.signal_loss_ticks <= self.recovery_wait_limit:
            self.mode = "SIGNAL_RECOVERY_WAIT"
            return curr_pos, f"SIGNAL_LOST_ATTEMPTING_RECOVERY ({self.signal_loss_ticks}/{self.recovery_wait_limit})"

        # Recovery failed -> Activate SAFETY_BACKUP 10-minute path retrace
        if self.mode != "SAFETY_BACKUP":
            self.mode = "SAFETY_BACKUP"
            self.retrace_queue = self.vault.get_10min_retrace_waypoints(window_sec=600.0)
            # Skip current cell if it's at the top of the breadcrumb stack
            if self.retrace_queue and self.retrace_queue[0] == curr_pos:
                self.retrace_queue.pop(0)

        if self.retrace_queue:
            next_backtrack_pos = self.retrace_queue.pop(0)
            return next_backtrack_pos, f"SAFETY_BACKUP_RETRACING_10MIN_PATH -> {next_backtrack_pos}"

        return curr_pos, "SAFETY_BACKUP_ORIGIN_REACHED_BEACON_ACTIVE"