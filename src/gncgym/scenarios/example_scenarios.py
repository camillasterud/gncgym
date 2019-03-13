import numpy as np
from numpy import pi, sin, cos, arctan2
from gncgym.base_env.base import BaseShipScenario, NS, OBST_RANGE, LOS_DISTANCE
from gncgym.parametrised_curves import RandomLineThroughOrigin, RandomCurveThroughOrigin, ParamCircle, ParamLine
from gncgym.base_env.objects import Vessel2D, AUV2D, StaticObstacle, DynamicObstacle, distance, MAX_SURGE, CROSS_TRACK_TOL, SURGE_TOL
from gncgym.simulator.angle import Angle
from gncgym.utils import distance, rotate


class ExampleScenario(BaseShipScenario):
    def generate(self, rng):
        self.path = RandomLineThroughOrigin(rng, length=500)
        x, y = self.path(0)
        angle = self.path.get_angle(0)
        self.speed = 4

        self.ship = Vessel2D(angle, x, y)

        self.static_obstacles.append(
            StaticObstacle(position=self.path(100), radius=10, color=(0.6, 0, 0))
        )

        self.dynamic_obstacles.append(
            DynamicObstacle(self.path, speed=4, init_s=50)
        )


class StraightPathScenario(BaseShipScenario):
    def generate(self, rng):
        self.path = RandomLineThroughOrigin(rng, length=500)
        self.speed = 4

        x, y = self.path(0)
        angle = self.path.get_angle(0)
        x += 2*(rng.rand()-0.5)
        y += 2*(rng.rand()-0.5)
        angle += 0.1*(rng.rand()-0.5)
        self.ship = Vessel2D(angle, x, y)


class CurvedPathScenario(BaseShipScenario):
    def __init__(self, linearising_feedback=True):
        self.linFB = linearising_feedback
        super().__init__()

    def generate(self, rng):
        L = 400
        a = 2*np.pi*(rng.rand()-0.5)
        self.path = RandomCurveThroughOrigin(rng, start=((L*cos(a), L*sin(a))))
        self.speed = 4

        x, y = self.path(0)
        angle = self.path.get_angle(0)
        x += 2*(rng.rand()-0.5)
        y += 2*(rng.rand()-0.5)
        angle += 0.1*(rng.rand()-0.5)
        self.ship = Vessel2D(angle, x, y, linearising_feedback=self.linFB)


class CircularPathScenario(BaseShipScenario):
    def generate(self, rng):
        self.path = ParamCircle((0, 0), 300)
        self.speed = 4

        x, y = self.path(0)
        angle = self.path.get_angle(0)
        x += 2*(rng.rand()-0.5)
        y += 2*(rng.rand()-0.5)
        angle += 0.1*(rng.rand()-0.5)
        self.ship = Vessel2D(angle, x, y)


class StraightPathOvertakingScenario(BaseShipScenario):
    def generate(self, rng):
        L = 400
        a = 2*np.pi*(rng.rand()-0.5)
        self.path = RandomLineThroughOrigin(rng, start=((L*cos(a), L*sin(a))))
        self.speed = 4

        x, y = self.path(0)
        angle = self.path.get_angle(0)
        x += 2*(rng.rand()-0.5)
        y += 2*(rng.rand()-0.5)
        angle += 0.1*(rng.rand()-0.5)

        self.dynamic_obstacles.append(DynamicObstacle(self.path, speed=2, init_s=20))
        self.ship = Vessel2D(angle, x, y)


class CurvedPathOvertakingScenario(BaseShipScenario):
    def generate(self, rng):
        L = 400
        a = 2*np.pi*(rng.rand()-0.5)
        self.path = RandomCurveThroughOrigin(rng, start=((L*cos(a), L*sin(a))))
        self.speed = 4

        x, y = self.path(0)
        angle = self.path.get_angle(0)
        x += 2*(rng.rand()-0.5)
        y += 2*(rng.rand()-0.5)
        angle += 0.1*(rng.rand()-0.5)

        self.dynamic_obstacles.append(DynamicObstacle(self.path, speed=2, init_s=20))
        self.ship = Vessel2D(angle, x, y)


class StraightPathShipCollisionScenario(BaseShipScenario):
    def generate(self, rng):
        # L = 400
        # a = 2*np.pi*(rng.rand()-0.5)
        self.path = RandomLineThroughOrigin(rng, length=400)
        self.speed = 4
        x, y = self.path(0)
        angle = self.path.get_angle(0)
        x += 2*(rng.rand()-0.5)
        y += 2*(rng.rand()-0.5)
        angle += 0.1*(rng.rand()-0.5)
        self.ship = Vessel2D(angle, x, y)

        self.dynamic_obstacles.append(DynamicObstacle(reversed(self.path), speed=2, init_s=20))


class CurvedPathShipCollisionScenario(BaseShipScenario):
    def generate(self, rng):
        L = 400
        a = 2*np.pi*(rng.rand()-0.5)
        self.path = RandomCurveThroughOrigin(rng, start=((L*cos(a), L*sin(a))))
        self.speed = 4

        x, y = self.path(0)
        angle = self.path.get_angle(0)
        x += 2*(rng.rand()-0.5)
        y += 2*(rng.rand()-0.5)
        angle += 0.1*(rng.rand()-0.5)
        self.ship = Vessel2D(angle, x, y)

        self.dynamic_obstacles.append(DynamicObstacle(reversed(self.path), speed=2, init_s=20))


class CurvedPathStaticObstacles(BaseShipScenario):
    def generate(self, rng):
        L = 400
        a = 2 * np.pi * (rng.rand() - 0.5)
        self.path = RandomCurveThroughOrigin(rng, start=((L * cos(a), L * sin(a))))
        self.speed = 4

        x, y = self.path(0)
        angle = self.path.get_angle(0)
        x += 2*(rng.rand()-0.5)
        y += 2*(rng.rand()-0.5)
        angle += 0.1*(rng.rand()-0.5)
        self.ship = Vessel2D(angle, x, y, linearising_feedback=False)

        for i in range(10):
            self.static_obstacles.append(StaticObstacle(
                self.path(0.9*self.path.length*(rng.rand() + 0.1)).flatten() + 100*(rng.rand(2)-0.5), radius=10*(rng.rand()+0.5) ))


class CurvedPathStaticDynamicObstacles(BaseShipScenario):
    def generate(self, rng):
        L = 400
        a = 2 * np.pi * (rng.rand() - 0.5)
        self.path = RandomCurveThroughOrigin(rng, start=((L * cos(a), L * sin(a))))
        self.speed = 4

        x, y = self.path(0)
        angle = self.path.get_angle(0)
        x += 2*(rng.rand()-0.5)
        y += 2*(rng.rand()-0.5)
        angle += 0.1*(rng.rand()-0.5)
        self.ship = Vessel2D(angle, x, y)

        straightpath = ParamLine(startpoint=self.path(0).flatten(), endpoint=self.path.get_endpoint().flatten())

        for i in range(rng.randint(5, 20)):
            self.static_obstacles.append(StaticObstacle(
                self.path(0.9*self.path.length*(rng.rand() + 0.1)).flatten() + 100*(rng.rand(2)-0.5), radius=10*(rng.rand()+0.5) ))

        for i in range(rng.randint(-10, 3)):
            init_s = 0.9 * self.path.length * (rng.rand() + 0.1)
            speed = (rng.rand()+1/6)*6
            p = rng.choice([self.path, reversed(self.path), straightpath, reversed(straightpath)])
            self.dynamic_obstacles.append(DynamicObstacle(path=p, speed=speed, init_s=init_s))


class StraightPathScenarioAUV(BaseShipScenario):
    def generate(self, rng):
        self.path = RandomLineThroughOrigin(rng, length=500)
        self.speed = 1.2

        x, y = self.path(0)
        angle = self.path.get_angle(0)
        x += 2*(rng.rand()-0.5)
        y += 2*(rng.rand()-0.5)
        angle += 0.1*(rng.rand()-0.5)
        self.ship = AUV2D(angle, x, y)

    def step_reward(self, action, obs, ds):
        done = False
        x, y = self.ship.position
        step_reward = 0

        # Living penalty
        # step_reward -= 0.001  # TODO Increase living penalty

        if not done and self.reward < -50:
            done = True

        if not done and abs(self.s - self.path.length) < 1:
            done = True

        if not done and distance(self.ship.position, self.path.get_endpoint()) < 20:
            done = True
            # step_reward += 50

        if not done:  # Reward progress along path, penalise backwards progress
            step_reward += ds/2

        if not done:  # Penalise cross track error if too far away from path
            # state_error = obs[:6]
            # step_reward += (0.2 - np.clip(np.linalg.norm(state_error), 0, 0.4))/100
            # heading_err = state_error[2]
            # surge_err = state_error[3]
            # TODO Punish for facing wrong way / Reward for advancing along path

            surge_error = obs[0]
            cross_track_error = obs[2]

            # step_reward -= abs(cross_track_error)*0.1
            # step_reward -= max(0, -surge_error)*0.1

            step_reward -= abs(cross_track_error)*0.5 + max(0, -surge_error)*0.5

            # step_reward -= (max(0.1, -obs[0]) - 0.1)*0.3
            # dist_from_path = np.sqrt(x_err ** 2 + y_err ** 2)
            # path_angle = self.path.get_angle(self.s)
            # If the reference is pointing towards the path, don't penalise
            # if dist_from_path > 0.25 and sign(float(Angle(path_angle - self.ship.ref[1]))) == sign(y_err):
            #     step_reward -= 0.1*(dist_from_path - 0.25)

        return done, step_reward


class CurvedPathScenarioAUV(BaseShipScenario):
    def __init__(self):
        super().__init__()

    def generate(self, rng):
        L = 400
        a = 2*np.pi*(rng.rand()-0.5)
        self.path = RandomCurveThroughOrigin(rng, start=((L*cos(a), L*sin(a))))
        self.speed = 1.2

        x, y = self.path(0)
        angle = self.path.get_angle(0)
        x += 2*(rng.rand()-0.5)
        y += 2*(rng.rand()-0.5)
        angle += 0.1*(rng.rand()-0.5)
        self.ship = AUV2D(angle, x, y)
    
    def step_reward(self, action, obs, ds):
        done = False
        x, y = self.ship.position
        step_reward = 0

        # Living penalty
        # step_reward -= 0.001  # TODO Increase living penalty

        if not done and self.reward < -50:
            done = True

        if not done and abs(self.s - self.path.length) < 1:
            done = True

        if not done and distance(self.ship.position, self.path.get_endpoint()) < 20:
            done = True
            # step_reward += 50

        if not done:  # Reward progress along path, penalise backwards progress
            step_reward += ds/2


        if not done:  # Penalise cross track error if too far away from path
            # state_error = obs[:6]
            # step_reward += (0.2 - np.clip(np.linalg.norm(state_error), 0, 0.4))/100
            # heading_err = state_error[2]
            # surge_err = state_error[3]
            # TODO Punish for facing wrong way / Reward for advancing along path

            surge_error = obs[0]
            cross_track_error = obs[2]

            # step_reward -= abs(cross_track_error)*0.1
            # step_reward -= max(0, -surge_error)*0.1

            step_reward -= abs(cross_track_error)*0.5 + max(0, -surge_error)*0.5

            # step_reward -= (max(0.1, -obs[0]) - 0.1)*0.3
            # dist_from_path = np.sqrt(x_err ** 2 + y_err ** 2)
            # path_angle = self.path.get_angle(self.s)
            # If the reference is pointing towards the path, don't penalise
            # if dist_from_path > 0.25 and sign(float(Angle(path_angle - self.ship.ref[1]))) == sign(y_err):
            #     step_reward -= 0.1*(dist_from_path - 0.25)
        return done, step_reward


class CurvedPathStaticObstaclesAUV(BaseShipScenario):
    def generate(self, rng):
        from gym.utils import seeding
        if self.config["randomness"]:
            rng = self.np_random
            rng_path = rng
        else:
            rng, seed = seeding.np_random(5)
            rng_path = None
        L = 400
        a = 2 * np.pi * (rng.rand() - 0.5)
        self.path = RandomCurveThroughOrigin(rng=rng_path, start=((L * cos(a), L * sin(a))))
        self.speed = 1.5
        self.max_speed = 1.8

        x, y = self.path(0)
        angle = self.path.get_angle(0)
        x += 2*(rng.rand()-0.5)
        y += 2*(rng.rand()-0.5)
        angle += 0.1*(rng.rand()-0.5)
        self.ship = AUV2D(angle, x, y, linearising_feedback=False)
        num_obs = self.config["num_obstacles"]
        self.num_sectors = self.config["num_sectors"]

        for i in range(num_obs):
            self.static_obstacles.append(StaticObstacle(
                self.path(0.9*self.path.length*(rng.rand() + 0.1)).flatten() + 100*(rng.rand(2)-0.5), radius=10*(rng.rand()+0.5) ))

    def step_reward(self, action, obs, ds):
        done = False
        step_reward = 0

        if not done and self.reward < -300:
            done = True

        if not done and abs(self.s - self.path.length) < 1:
            done = True

        #for o in self.static_obstacles + self.dynamic_obstacles:
        #    if not done and distance(self.ship.position, o.position) < self.ship.radius + o.radius:
        #        done = True
        #        step_reward += self.config["reward_collision"]
        #        break

        if not done and distance(self.ship.position, self.path.get_endpoint()) < 20:
            done = True

        if not done:
            step_reward += ds*self.config["reward_ds"]

        for i, slot in self.active_static.items():
            closeness = obs[NS + 2*slot + 1]
            step_reward += self.config["reward_closeness"]*closeness**2

        if not done: 

            surge_error = obs[0] - self.speed/self.max_speed
            cross_track_error = obs[2]

            step_reward += abs(cross_track_error)*self.config["reward_cross_track_error"]
            step_reward += max(0, -surge_error)*self.config["reward_surge_error"]

        return done, step_reward

    def navigate(self, state=None):
        LOS_DISTANCE = 25
        OBST_RANGE = 150
        if state is None:
            state = self.ship.state.flatten()

        self.update_closest_obstacles()

        # TODO Try lookahead vector instead of closest point
        closest_point = self.path(self.s).flatten()
        closest_angle = self.path.get_angle(self.s)
        target = self.path(self.s + LOS_DISTANCE).flatten()
        target_angle = self.path.get_angle(self.s + LOS_DISTANCE)

        # State and path errors
        surge_error = self.speed - state[3]
        heading_error = float(Angle(target_angle - state[2]))
        cross_track_error = rotate(closest_point - self.ship.position, -closest_angle)[1]
        target_dist = distance(self.ship.position, target)

        # Construct observation vector
        obs = np.zeros((NS + self.num_sectors,))


        obs[0] = np.clip((state[3]**2 + state[4]**2) /self.max_speed, 0, 1)
        obs[1] = np.clip(heading_error / pi, -1, 1)
        obs[2] = np.clip(cross_track_error / LOS_DISTANCE, -1, 1)
        obs[3] = np.clip(self.last_action[0], -1, 1)
        obs[4] = np.clip(self.last_action[1], 0, 1)

        distance_vecs = {i: (self.ship.position - obst.position) for i, obst in enumerate(self.static_obstacles)}
        for obsti, obst in enumerate(self.static_obstacles):
            vec = distance_vecs[obsti]
            dist = np.linalg.norm(vec)
            if dist < OBST_RANGE:
                ang = (float(Angle(arctan2(vec[1], vec[0]) - self.ship.angle)) + pi/2)/pi
                if 0 <= ang < 1:
                    closeness = 1 - np.clip((dist - self.ship.radius - obst.radius) / OBST_RANGE, 0, 1)
                    if obs[np.floor(ang*self.num_sectors)] > closeness:
                        obs[NS + np.floor(ang*self.num_sectors)] = closeness

        return obs