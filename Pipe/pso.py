# -*- coding: utf-8 -*-

"""
pso.py
Functions related to custom implementation of PSO algorithm.
"""
__author__      = "JORGE H. CARDENAS"
__copyright__   = "2024,2025"
__version__   = "1.0"


from numpy.core.records import array

import os

import numpy as np
import importlib

import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate
from sklearn.metrics import mean_squared_error

from numpy.random import seed
from numpy.random import randn

import torch
import torch.nn as nn
#seed control
#seed(1)



class Particle:
    """
    Particle class enabling the creation of multiple and dynamic number of particles.
    """
    id_ = 0
    values_array = []

    def __init__(self, id ):
        self.id_=id

        

    def random_array(self, array_size):
        self.values_array = randn(array_size) # 0.5+0.5*randn(array_size)

    def fill_zeros_array(self, array_size):
       self.values_array = np.zeros(array_size)
    
    def reset_xParticles_id_counter(self):
        self.id_ = 0


class Swarm:
    
    phiv=0.5

    particles=[] # array with all particles part of the swar
    x_particles=[] # array with all particles part of the swar
    best_index=0 #best integer describing the index corresponding the best particle in 
                #particles array
    vmax = []
    velocidades = []
    gbest = 0  #global  best
    pbest =[]
    pg = []
    variables_number=0

    def __init__(self, particles_number, variables_number, var_max, var_min):
        #swarm variables
        self.particles_number = particles_number
        self.variables_number = variables_number

        #Set variations
        
        self.var_max = np.array(var_max)
        self.var_min = np.array(var_min)
        self.particles = []
        self.velocidades = np.zeros([self.particles_number,self.variables_number])
        self.pbest = np.zeros(self.particles_number)
        self.pg = np.zeros(self.variables_number)

    
    """Create particles swarm"""
    def create(self):
        
        self.particles = [Particle(i) for i in range(self.particles_number )]
        self.x_particles = [Particle(i) for i in range(self.particles_number  )]
        
        print("particulas creadas:"+str(len(self.particles)))

        interval_array=np.array(self.var_max) - np.array(self.var_min)
        self.vmax = interval_array * 0.6

        for particle in self.particles:
            #Generate random array for each particle
            particle.random_array(self.variables_number)
            #Scale the random values
            particle.values_array = particle.values_array*(self.var_max-self.var_min) + self.var_min


    def nuevas_particulas(self,particulas_ant, pi_best, pg, vel_anterior, iteration):
    #     #Particulas => xi(t-1)
    #     #pi => individual optimal position!?
    #     # x son particulas que vienen definidas desde la creacion del enjambre
    #     # Solo se van actualizando en el transcurso de las iteraciones

        [item.fill_zeros_array(self.variables_number) for item in self.x_particles]#llenar de ceros las particulas x

        vel = np.zeros([self.particles_number, self.variables_number])#llenar de ceros la variable velocidad
        
        #Cambio dinámico de la inercia
        phi = 0.5-((0.5-0.35)*iteration/(30) )# adaptive inertia method
        self.phiv =phi
        
        phi1 = 2.4 #valores que se pueden revisar. Seguir el valor el mejor fit propio
        phi2 = 1.3 #esto va valores componen self-knowledge.  seguir el mejor fit global
        damping = 0.95 #este damping se utiliza cuando las particulas tocan los limitesmáximos y mínimos.

        for i in range(self.particles_number):
            
            rand = np.random.uniform(low=0., high=1.0) #valores entre 0 y 1
            rand2 = np.random.uniform(low=0., high=1.0) #valores entre 0 y 1

            particula_anterior=particulas_ant[i]

            for idx,dimension in enumerate(particula_anterior.values_array):

                """ 
                Calcula la velocidad de la particula i, dada su pi, pg y su velocidad
                 y posicion anterior 

                 pi => una particula especifica
                pbest=> su fitness value

                 pg => cualquier particula con la mejor solucion
                 gbest=> global bets

                 vi => velocidad anterior
                """

                """lo importante es la realción entre phi1 y phi2 , si es grande la velocidad de convergencia es alta
                si la relacion es pequeña, la velocidad es baja"""

       

                ###inercia +  atraccion a la mejor posicion de la particula i +  atraccion a la mejor posición global
                ###

                vel[i][idx] = phi * (vel_anterior[i][idx]) + phi1 * rand * ((pi_best[i].values_array)[idx] - dimension) + \
                     phi2* rand2 * (pg[idx] - dimension)

                #Limiting velocity when too large
                if np.abs(vel[i][idx]) > self.vmax[idx]:
                    signo = np.sign(vel[i][idx])
                    vel[i][idx] =  self.vmax[idx]*signo
                    

                #Calculating new paticles
                self.x_particles[i].values_array[idx] = (particulas_ant[i].values_array[idx] + vel[i][idx]).round(decimals=4, out=None) ## xi(t-1)+vi( 

                """In this part we apply a bounce tecnique in the wall defined
                by the limits of max and min values for dimensions"""       
                if self.x_particles[i].values_array[idx] > self.var_max[idx]:
                    
                    
                    vel[i][idx]= vel[i][idx]*damping

                    self.x_particles[i].values_array[idx] = (self.var_max[idx]-np.abs(vel[i][idx])).round(decimals=4, out=None) ## xi(t-1)+vi(t)
                    

                elif self.x_particles[i].values_array[idx] < self.var_min[idx]:
                   
                    vel[i][idx]= vel[i][idx]*damping

                    self.x_particles[i].values_array[idx] = (self.var_min[idx]+np.abs(vel[i][idx]) ).round(decimals=4, out=None) ## xi(t-1)+vi(t)
                    
                else:
                    self.x_particles[i].values_array[idx] = (particulas_ant[i].values_array[idx] + vel[i][idx]).round(decimals=4, out=None) ## xi(t-1)+vi( 

                #This sections is used to force a=B*2
                # if idx==global_.A_dimension_index+1:
                #    self.x_particles[i].values_array[idx] = (self.x_particles[i].values_array[idx-1])*0.5
           
        return self.x_particles,vel



    def get_particle_best_fit(self, pi):

        index_pg = np.argmin(self.pbest) #toma el indice del particle best entre todas las particulas
        self.best_index = index_pg
        #print("get particle best pg="+str(pi[index_pg].values_array))
        self.pg = pi[index_pg].values_array # seleccionar la mejor posicion-particula del array de particulas
        self.gbest = np.min(self.pbest)  #best global fitness 
        return index_pg

#Fitness func how close a given solution is to the optimum solution
def fitness(predicted,truth,iter):
    
    #get results
    Y_true = truth
    Y_pred = predicted
    # Calculation of Mean Squared Error (MSE)
    loss = nn.MSELoss()

    fitness =  loss(Y_pred,Y_true)  #Esta variable quedará para hacer una funcion compuesta en el futuro
    #fitness = fitness.round(decimals=2, out=None)

    print("fitness="+str(fitness.detach().item()))
    
 
    return fitness



