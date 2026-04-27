from mpl_toolkits import mplot3d
import numpy as np 
import matplotlib.pyplot as plt
from missile import Missile
from interceptor import Interceptor
from target_location import TargetLocation

fig = plt.figure()
plt.title("3D Missile Interception")    
ax = fig.add_subplot(projection = '3d')

target_location = TargetLocation(np.array([5,0,0]))
interceptor = Interceptor(np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]))
missile = Missile(np.array([10,10,10]), np.array([0,0,0]), np.array([0,0,0]))

ax.scatter(target_location.location[0],target_location.location[1],target_location.location[2],color='b',label='target')
ax.scatter(interceptor.starting_point[0],interceptor.starting_point[1],interceptor.starting_point[2],color='r',label='interceptor')
ax.scatter(missile.starting_point[0],missile.starting_point[1],missile.starting_point[2],color='g',label='missile')
ax.plot([target_location.location[0]],[target_location.location[1]],[target_location.location[2]])
ax.plot([interceptor.starting_point[0]],[interceptor.starting_point[1]],[interceptor.starting_point[2]])
ax.plot([missile.starting_point[0]],[missile.starting_point[1]],[missile.starting_point[2]])

plt.legend()
plt.show()