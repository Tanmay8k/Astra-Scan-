import numpy as np
from ai_trainer import train_from_csv, train_from_image_folder
from database_vault import SouthPoleVault
from safety_autonomous_engine import AutonomousSafetyEngine


class SouthPoleRoverBackend:
    def __init__(self, grid_size: int = 48):
        self.grid_size = grid_size
        self.vault = SouthPoleVault()
        self.engine = AutonomousSafetyEngine(self.vault)
        self.rover_pos = (22, 12)
        self.heading = "NORTH"
        self.signal_pct = 92.0

    def train_custom_database(self, path: str, is_image_folder: bool = True) -> dict:
        """Call this to train the AI on your own Lunar South Pole images or CSV database."""
        if is_image_folder:
            stats = train_from_image_folder(path)
        else:
            stats = train_from_csv(path)
        # Reload freshly trained model into the active autonomous engine
        self.engine = AutonomousSafetyEngine(self.vault)
        return stats

    def step_mission(
        self,
        dem: np.ndarray,
        thermal_k: np.ndarray,
        camera_frame: np.ndarray,
        simulated_signal_pct: float | None = None,
    ) -> dict:
        """
        Single backend tick:
        1. Runs 60m forward AI lookahead prediction.
        2. Checks Earth signal; if lost & unrecovered, retraces 10-min path in SAFETY_BACKUP.
        3. Saves Lunar South Pole image frame + breadcrumb to vault.
        """
        if simulated_signal_pct is not None:
            self.signal_pct = simulated_signal_pct

        # 1. Forward Lookahead Prediction (60m ahead)
        lookahead = self.engine.predict_forward_corridor(
            dem, thermal_k, self.rover_pos, self.heading, lookahead_steps=6
        )

        # 2. Check signal & execute SAFETY_BACKUP 10-min retrace if needed
        new_pos, comms_status = self.engine.evaluate_comms_and_backup(
            self.rover_pos, self.signal_pct
        )

        if self.engine.mode == "SAFETY_BACKUP":
            self.rover_pos = new_pos
            # Simulate signal gradually recovering as rover retraces out of crater shadow
            if len(self.engine.retrace_queue) <= 1:
                self.signal_pct = 85.0
        elif self.engine.mode == "AUTONOMOUS_EXPLORE":
            # Move 1 step along the AI-cleared corridor lane
            center_wp = lookahead["corridors"]["CENTER_LANE"]["waypoints"][0]["pos"]
            left_wp = lookahead["corridors"]["LEFT_LANE"]["waypoints"][0]["pos"]
            right_wp = lookahead["corridors"]["RIGHT_LANE"]["waypoints"][0]["pos"]

            if lookahead["recommended_action"] == "PROCEED_CENTER_NOMINAL":
                self.rover_pos = center_wp
            elif lookahead["recommended_action"] == "STEER_LEFT_AVOID_HAZARD":
                self.rover_pos = left_wp
            elif lookahead["recommended_action"] == "STEER_RIGHT_AVOID_HAZARD":
                self.rover_pos = right_wp

        r, c = self.rover_pos
        lat = round(90.0 - (r / (self.grid_size - 1)) * 180.0, 2)
        lon = round((c / self.grid_size) * 360.0 - 180.0, 2)
        curr_temp = float(thermal_k[r, c])
        center_pred = lookahead["corridors"]["CENTER_LANE"]["waypoints"][0]["prediction"]

        # 3. Record breadcrumb & save Lunar South Pole image to vault
        self.vault.record_breadcrumb(r, c, lat, lon, self.signal_pct, self.engine.mode)
        saved_img_path = self.vault.save_southpole_image(
            camera_frame,
            r,
            c,
            curr_temp,
            center_pred,
            signal_ok=(self.signal_pct >= 25.0),
        )

        return {
            "mode": self.engine.mode,
            "rover_pos": self.rover_pos,
            "coords": (lat, lon),
            "signal_pct": self.signal_pct,
            "comms_status": comms_status,
            "lookahead_advice": lookahead["recommended_action"],
            "lookahead_60m": lookahead["corridors"]["CENTER_LANE"]["waypoints"],
            "saved_southpole_image": saved_img_path,
        }


# Quick standalone verification test
if __name__ == "__main__":
    backend = SouthPoleRoverBackend()
    dummy_dem = np.random.uniform(0.1, 0.9, (48, 48))
    dummy_temp = 45.0 + dummy_dem * 220.0
    dummy_cam = np.random.randint(40, 210, (120, 160, 3), dtype=np.uint8)

    print("1. Normal Autonomous Explore + Forward Lookahead:")
    for _ in range(3):
        out = backend.step_mission(dummy_dem, dummy_temp, dummy_cam, simulated_signal_pct=90.0)
        print(f"   Pos={out['rover_pos']} | Advice={out['lookahead_advice']} | Saved={out['saved_southpole_image']}")

    print("\n2. Simulating Lunar South Pole Crater Shadow Signal Blackout:")
    for tick in range(5):
        out = backend.step_mission(dummy_dem, dummy_temp, dummy_cam, simulated_signal_pct=5.0)
        print(f"   Tick {tick+1}: Mode={out['mode']} | Pos={out['rover_pos']} | Status={out['comms_status']}")