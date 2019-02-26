from gym.envs.registration import register
from gym.error import Error as GymError
import gncgym.scenarios as scenarios


"""
Wrap in a try-catch block because gym complains about re-registering 
environments when running tests.
"""
try:
    register(
        id='shipExampleScenario-v0',
        entry_point='gncgym.scenarios.example_scenarios:ExampleScenario',
    )

    register(
        id='shipStraightPathFollowing-v0',
        entry_point='gncgym.scenarios.example_scenarios:StraightPathScenario',
    )

    register(
        id='shipCurvedPathFollowing-v0',
        entry_point='gncgym.scenarios.example_scenarios:CurvedPathScenario',
    )

    register(
        id='shipStraightPathFollowingWithOvertaking-v0',
        entry_point='gncgym.scenarios.example_scenarios:StraightPathOvertakingScenario',
    )

    register(
        id='shipCurvedPathFollowingWithOvertaking-v0',
        entry_point='gncgym.scenarios.example_scenarios:CurvedPathOvertakingScenario',
    )

    register(
        id='shipCurvedPathFollowingWithShipCollision-v0',
        entry_point='gncgym.scenarios.example_scenarios:CurvedPathShipCollisionScenario',
    )

    register(
        id='shipCurvedPathStaticObstacles-v0',
        entry_point='gncgym.scenarios.example_scenarios:CurvedPathStaticObstacles',
    )

    register(
        id='shipCurvedPathStaticDynamicObstacles-v0',
        entry_point='gncgym.scenarios.example_scenarios:CurvedPathStaticDynamicObstacles',
    )

    register(
        id='AUVStraightPathFollowing-v0',
        entry_point='gncgym.scenarios.example_scenarios:StraightPathScenarioAUV',
    )
    register(
        id='AUVCurvedPathFollowing-v0',
        entry_point='gncgym.scenarios.example_scenarios:CurvedPathScenarioAUV',
    )
    register(
        id='AUVCurvedPathStaticObstacles-v0',
        entry_point='gncgym.scenarios.example_scenarios:CurvedPathStaticObstaclesAUV',
    )

except GymError as e:
    print(e)

