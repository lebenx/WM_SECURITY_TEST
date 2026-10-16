import unittest
from wm_security.envs import Timer,Workbench,Resource,trajectory


class EnvironmentTests(unittest.TestCase):
    def test_expiry_boundary(self):
        env=Timer();trajectory(env,['activate','move','wait'])
        result=env.step('use')
        self.assertTrue(result['item_done'])
        self.assertFalse(result['active'])
        env=Timer();trajectory(env,['activate','move','wait','wait'])
        self.assertFalse(env.step('use')['item_done'])

    def test_power_cycle_invalidates_calibration(self):
        env=Workbench();trajectory(env,['power_on','calibrate','load','power_off','power_on'])
        self.assertFalse(env.step('process')['product_done'])

    def test_resources_never_negative(self):
        env=Resource();trajectory(env,['charge','load','move','move','move'])
        result=env.step('deliver')
        self.assertFalse(result['delivered'])
        self.assertEqual(result['energy'],2)


if __name__=='__main__': unittest.main()
