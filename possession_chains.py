import json
import os
import re

import numpy as np
import pandas as pd

# pass_end coordinate qualifiers ids: (x, y) = (140, 141) applicable for event type: 1
# shot_blocked coordinate qualifiers ids: (x, y) = (146, 147) applicable for event types: (13,14,15,16)


class possession_chains:
    def __init__(self, team_id=3):
        self.team_id = team_id
        self.possession_data = self.get_team_possession_data()

    def get_team_possession_data(self):
        possession_data = []
        for file in os.listdir("season83/"):
            if file.endswith(".json"):
                with open(f"season83/{file}") as f:
                    game_meta = json.load(f)
                with open(f"season83/events/{file}") as f:
                    game = json.load(f)
            if (
                game_meta["matches"][0]["homeTeamId"] == self.team_id
                or game_meta["matches"][0]["awayTeamId"] == self.team_id
            ):
                possession_data.append((game_meta, game))
        return possession_data

    def create_ball_trajectories(self):
        ball_trajectories = []
        ball_trajectories_dfs = []
        for game_meta, game in self.possession_data:
            for play in game:
                if self.team_in_possession(play):
                    ball_trajectory = self.get_ball_trajectory(play)
                    ball_trajectories.append(
                        np.array(ball_trajectory, dtype=np.float64)
                    )
                    df = pd.DataFrame(ball_trajectory, columns=["x", "y"])
                    df["start_eventId"] = play[0]["eventId"]
                    df["game_id"] = game_meta["matches"][0]["matchId"]
                    ball_trajectories_dfs.append(df)
        return ball_trajectories, ball_trajectories_dfs

    def team_in_possession(self, play):
        previuous_outcome = False
        possession_owner = None
        for event in play:
            outcome = event["outcome"]
            if (previuous_outcome and outcome) or (outcome and len(play) == 1):
                possession_owner = event["teamId"]
                break
            previuous_outcome = outcome

        if possession_owner != self.team_id:
            return False
        else:
            return True

    def get_ball_trajectory(self, play):
        ball_trajectory = []

        for event in play:
            ball_trajectory.append((event["x"], event["y"]))

            if event["eventTypeId"] == 1:
                x = [q["value"] for q in event["qualifiers"] if q["qualifierId"] == 140]
                y = [q["value"] for q in event["qualifiers"] if q["qualifierId"] == 141]

                if len(x) == len(y) == 1:
                    ball_trajectory.append((x[0], y[0]))

            elif event["eventTypeId"] in [13, 14, 15, 16]:
                x = [q["value"] for q in event["qualifiers"] if q["qualifierId"] == 146]
                y = [q["value"] for q in event["qualifiers"] if q["qualifierId"] == 147]

                if len(x) == len(y) == 1:
                    ball_trajectory.append((x[0], y[0]))

        return ball_trajectory
