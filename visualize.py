from mpl_toolkits import mplot3d
import numpy as np 
import matplotlib.pyplot as plt

fig = plt.figure()
plt.title("3D Missile Interception")
ax = fig.add_subplot(projection = '3d')


target_location = np.array([5,0,0])
interceptor_starting_point = np.array([0,0,0])
missile_starting_point= np.array([10,10,10])

ax.scatter(target_location[0],target_location[1],target_location[2],color='b',label='target')
ax.scatter(interceptor_starting_point[0],interceptor_starting_point[1],interceptor_starting_point[2],color='r',label='interceptor')
ax.scatter(missile_starting_point[0],missile_starting_point[1],missile_starting_point[2],color='g',label='missile')
ax.plot([target[0]],[target[1]],[target[2]])
ax.plot([interceptor[0]],[interceptor[1]],[interceptor[2]])
ax.plot([missile_starting_point[0]],[missile_starting_point[1]],[missile_starting_point[2]])

plt.legend()
plt.show()