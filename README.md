
# OpenAi Gym with CarRacing-v1 (unofficial)

This repo has improvements on the complexity for CarRacing-v1

## Improved

* Complexity of map (several types of intersections)
* Complexity of map (different numbers of lanes)
* Obstacles added
* More control over the agent
* Easy to modify the reward function

Here some images of some changes:

![junc](img/junc.png)
![t-junc](img/t.png)
![obstacle](img/obst.png)

### Maps

Some maps

![map](img/map1.png)
![map](img/map2.png)


## Improvements

This are some improvements of the environment, this allows configures each experiments depending on the objective


### Set the car in certain position in the map

`place_agent (position)` : this function will place the car in `position`


### Set intial speed of agent

`set_speed(speed)`: This function will set the initial of the car


### Ger random position in the track

`get_rnd_point_in_track(border)` : returns a random point in the track with the angle equal to the tile of the track, the x position can be randomly in the x (relative) axis of the tile, border=True make sure the x position is enough to make the car fit in the track, otherwise the point can be in the extreme of the track and two wheels will be outside the track
Returns: [beta, x, y]


## To Improve (by importance)

1. Do NOT compare first tile of main track with last tile of `self.track`
1. Get tile with certain conditions

   3. with certain angle (e.g. >40º)
   3. In an T-junction
   3. in an X-junction
   3. With obstacle in front

3. Get outside position
3. Detect change of line
3. Add obstacles
3. Add road lines
3. Change car when not racing
 
---

