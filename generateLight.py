import numpy as np
import matplotlib

import matplotlib.pyplot as plt

import math
from mpl_toolkits.mplot3d import Axes3D


class Rotation2point(object):

    def __init__(self,vec_a, vec_b):
        
        self.vec_a=vec_a/np.linalg.norm(vec_a)
        self.vec_b=vec_b/np.linalg.norm(vec_b)
        self.rotationMatrix=[]


    def calculate(self):

        v=np.cross(self.vec_a,self.vec_b)
        s=np.linalg.norm(v)
        c=np.dot(self.vec_a,self.vec_b)
        I=np.eye(3,dtype=np.float)
        v_m=np.matrix([[0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1],v[0],0]])

        self.rotationMatrix=I+v_m+v_m*v_m*(1/(1+c))
        return self.rotationMatrix

    def rotatePointsbyMatrix(self,pointset):
        
        pointset_norm=pointset/np.linalg.norm(pointset,axis=0)
        pointset_rot=self.rotationMatrix*pointset_norm
        pointset_rot_norm=pointset_rot/np.linalg.norm(pointset_rot,axis=0)
        return pointset_rot_norm

class LightGen(object):
    """
    generate lights which uniformly attribute in a unit sphere
    """
    
    def __init__(self, N, angle, isHalfRequired, lightzmargin,method="random"):
        assert N,"Light number is zero"
        self.method=method
        self.N=N
        self.angle=angle/np.linalg.norm(angle)
        self.LightNum=0
        self.L=[]
        self.x=[]
        self.y=[]
        self.z=[]
        self.fi=(np.sqrt(5)-1)/2
        self.isHalfRequired=isHalfRequired
        self.lightzmargin = lightzmargin
        self.direction_init=np.array([0,0,1])


    def generate(self):

        if self.method=="regular":
            self.L=self.genRegular()
        elif self.method=="random":
            self.L =self.genRandom()
        else:
            raise RuntimeError('No specific method : regular/random, input is '.join(self.method))

        if self.isHalfRequired:
            
            index=self.L[2]>0
            self.L = self.L[:,index]
        if self.lightzmargin is not None:
            index = np.where(np.abs(self.L[2])>self.lightzmargin)[0]
            self.L = self.L[:, index]

        _, self.LightNum=self.L.shape
        return self.L.transpose() 




    def genRegular(self):
        index = np.arange(1, self.N + 1)
        self.z=(2*index-1)/self.N-1
        self.x=np.sqrt(1-self.z*self.z)*np.cos(2*math.pi*index*self.fi)
        self.y=np.sqrt(1-self.z*self.z)*np.sin(2*math.pi*index*self.fi)

        return np.vstack((self.x,self.y,self.z))

    def genRandom(self):
        np.random.seed(128)
        u = np.random.random(size=int(self.N))
        np.random.seed(256)
        v = np.random.random(size=int(self.N))
        theta = 2 * math.pi * u
        phi = np.arccos(2 * v - 1)
        self.x = np.sin(theta) * np.sin(phi)
        self.y = np.cos(theta) * np.sin(phi)
        self.z = np.cos(phi)
        return np.vstack((self.x, self.y, self.z))

    def drawLight(self):
        ax = plt.subplot(111, projection='3d')
        ax.scatter(self.L[0], self.L[1], self.L[2], c='b')
        ax.set_zlabel('Z')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plt.show()

    def rotate(self, direction):
        self.new_direction=direction
        rotation_op=Rotation2point(self.direction_init,self.new_direction)
        rotation_op.calculate()
        self.L=rotation_op.rotatePointsbyMatrix(self.L)

    def run(self):
        self.generate()
        if not (self.angle == self.direction_init).all():
            self.rotate(self.angle)

        self.drawLight()
        return self.L.T




if __name__ == '__main__':
    num_lights = 24
    generator=LightGen(num_lights*4,np.array([0,0,1]),True, 0.5, "regular")
    generator.run()

    light_ins = np.random.random(int(num_lights))     
    # plt.plot(light_ins)
    # plt.show()
    L_out = light_ins[:, np.newaxis] * generator.L.T

    


    
