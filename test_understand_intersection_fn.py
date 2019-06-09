import numpy as np

from gym.envs.box2d.car_racing import CarRacing

if __name__ == '__main__':
    env = CarRacing(load_tracks_from="/hdd/Documents/HRL/tracks")
    env.tracks_df = env.tracks_df[(env.tracks_df['x'])|(env.tracks_df['t'])]
    repeat = 1000
    for i in range(repeat):
        print("%i of %i" % (i,repeat))
        env.reset()
        dictionary= env.understand_intersection(
                np.random.choice(np.where(env.info['intersection_id'] != -1)[0]),
                1)

        # Checking that only one value can be None
        print(dictionary)
        if list(dictionary.values()).count(None) > 1: 
            print("breaking, more than 1 None")
            break

        for val in dictionary.values():
            if val is not None and type(val) is not list and len(val) != 2:
                print("Breaking, there is some value that \
                        is not a list or a different kind of list")
                break
        print("")
