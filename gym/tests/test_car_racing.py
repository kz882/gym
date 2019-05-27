import pytest
from gym.envs.box2d.car_racing import CarRacing

class TestCarRacing(object):
    def test_one_track(self):
        env = CarRacing()

        # Tracks should not exist before any reset
        with pytest.raises(AttributeError):
            env.tracks

        env.reset()
        assert len(env.tracks) == 1

        env.close()

    def test_two_tracks(self):
        env = CarRacing(num_tracks=2)

        env.reset()
        assert len(env.tracks) == 2

        env.close()

    def test_screenshot(self,tmpdir):
        # TODO use tempdir to save screenshots
        pass
