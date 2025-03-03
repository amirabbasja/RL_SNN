# Explanation

Here we use gymnasium's LuanrLander-v3 environment. According to the documentation, we have four actions in the action space, 0 (Do nnothing), 1 (Fire left orientation engine), 2 (Fire main engine) , 3 (Fire right orientation engine)

Also the observation space is a ndarray with shape (8,), It is as follows:

1, 2: It's (x,y) coordinates. The landing pad is always at coordinates (0,0).
3, 4: It's linear velocities (xDot,yDot).
5: It's angle Theta.
6: It's angular velocity thetaDot
7, 8: Two booleans l and r that represent whether each leg is in contact with the ground or not.
Rewards:

is increased/decreased the closer/further the lander is to the landing pad.
is increased/decreased the slower/faster the lander is moving.
is decreased the more the lander is tilted (angle not horizontal).
is increased by 10 points for each leg that is in contact with the ground.
is decreased by 0.03 points each frame a side engine is firing.
is decreased by 0.3 points each frame the main engine is firing.
The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively. An episode is considered a solution if it scores at least 200 points.

Episode end: The episode finishes if:

The lander crashes (the lander body gets in contact with the moon);
The lander gets outside of the viewport (x coordinate is greater than 1);
The lander is not awake. From the Box2D docs, a body which is not awake is a body which doesn’t move and doesn’t collide with any other body