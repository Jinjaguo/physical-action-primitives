"""Matched visible sphere, two explicit unobserved damping conditions."""
from mani_skill.envs.tasks.tabletop.roll_ball import RollBallEnv
from mani_skill.utils.registration import register_env

@register_env('PairedDampingRollBall-v1',max_episode_steps=10000)
class PairedRollBall(RollBallEnv):
    def __init__(self,*args,damping_low=0.,damping_high=8.,**kwargs):
        self.damping_values=(float(damping_low),float(damping_high))
        super().__init__(*args,**kwargs)

    def _load_scene(self,options):
        super()._load_scene(options)
        self.actual_damping=[]
        for i,body in enumerate(self.ball._bodies):
            value=self.damping_values[i%2]
            body.set_linear_damping(value);body.set_angular_damping(value)
            self.actual_damping.append([body.get_linear_damping(),body.get_angular_damping()])
