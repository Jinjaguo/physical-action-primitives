"""Single-lane-per-parent RollBall damping intervention environment."""
from mani_skill.envs.tasks.tabletop.roll_ball import RollBallEnv
from mani_skill.utils.registration import register_env


@register_env("SequentialDampingRollBall-v1", max_episode_steps=10000)
class SequentialDampingRollBall(RollBallEnv):
    def __init__(self, *args, preparation_damping=4.0, **kwargs):
        self.preparation_damping = float(preparation_damping)
        super().__init__(*args, **kwargs)

    def _load_scene(self, options):
        super()._load_scene(options)
        self.set_lane_damping([self.preparation_damping] * len(self.ball._bodies))

    def set_lane_damping(self, values):
        values = [float(value) for value in values]
        if len(values) != len(self.ball._bodies):
            raise ValueError("one damping value is required for each physical parent lane")
        for body, value in zip(self.ball._bodies, values, strict=True):
            body.set_linear_damping(value)
            body.set_angular_damping(value)

    def get_lane_damping(self):
        return [[float(body.get_linear_damping()), float(body.get_angular_damping())] for body in self.ball._bodies]
