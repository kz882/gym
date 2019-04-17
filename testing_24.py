from gym.envs.box2d.car_racing import CarRacing

if __name__ == '__main__':
    env = CarRacing()
    repeat = int(1.5e6)
    for i in range(repeat):
        print("%i of %i" % (i,repeat))
        env.reset()
        print("")
