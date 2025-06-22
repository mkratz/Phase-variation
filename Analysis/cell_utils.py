import numpy as np 
import pandas as pd
import random
import scipy.stats as st
from tqdm import tqdm

class Cell:
    def __init__(self,state_circ='yellow',state_counter=0,label='mother',age=0,arrival_times=[],trans_kwargs=None,spawn_kwargs=None):
        self.state_circ = state_circ # circuit state
        self.state_counter = state_counter # keeps track of how long been in yellow or blue state
        self.label = label
        self.trans_kwargs = trans_kwargs
        self.spawn_kwargs = spawn_kwargs
        self.age = age
        self.arrival_times = arrival_times
        
    def transition(self,p_ytob,p_btoy):
        # calculates if a transition occurs given current state of cell
        if self.state_circ == 'yellow':
            cx = str(np.random.choice(['yellow','blue'],size=1,p=[1-p_ytob,p_ytob])[0])
            self.state_circ = cx
            if cx != 'yellow':
                self.arrival_times.append(['to blue',self.state_counter])
                self.state_counter = 0

        elif self.state_circ == 'blue':
            cx = str(np.random.choice(['blue','yellow'],size=1,p=[1-p_btoy,p_btoy])[0])
            self.state_circ = cx
            if cx != 'blue':
                self.arrival_times.append(['to yellow',self.state_counter])
                self.state_counter = 0
    
    def spawn(self,div_constant,):
        # method for spawning N new cell objects from existing cell objects
        cell_list = []
        for n in np.arange(0,div_constant):
            cell_list.append(Cell(self.state_circ,self.state_counter,'daughter',0,[],self.trans_kwargs,self.spawn_kwargs))
        return cell_list
    
    def run_full_timestep(self):
        # basically lumps together transition and counter methods
        self.age+=1 
        self.transition(*self.trans_kwargs)
        self.state_counter+=1
        
class cell_env:
    def __init__(self,max_pop=1000):
        self.running_list = [] # list that contains all cells for a given time step
        self.max_pop = max_pop # limit size of population tracking
        self.history = pd.DataFrame(columns=['yellow','blue','daughters']) # df that tracks pop info
        self.meta = None
        
    ### Functions for spawning initial cell pop    
    def initialize_cells_det(self,pop_seed=[5,5],trans_kwargs=None,spawn_kwargs=None):
        # Spawn N inactive cells
        yellow_count,blue_count = pop_seed
        for n in range(0,yellow_count):
            self.running_list.append(Cell(state_circ='yellow',arrival_times=[],trans_kwargs=trans_kwargs,spawn_kwargs=spawn_kwargs))
        for n in range(0,blue_count):
            self.running_list.append(Cell(state_circ='blue',arrival_times=[],trans_kwargs=trans_kwargs,spawn_kwargs=spawn_kwargs))
            
    ### Functions that happen each time step        
    def run_trans_counter(self):
        # run timestep for each cell (transition,counter_update)
        for cell in self.running_list:
            cell.run_full_timestep()
        
    def run_spawn(self):
        # make next generation via running spawn for each cell
        temp_run = [] # holder for newly generated cells
        for cell in self.running_list:
            temp_run+=cell.spawn(*cell.spawn_kwargs)
        self.running_list+=temp_run
    
    def cull(self):
        # if max_pop exceeded by spawning, randomly select N=max_pop cells from spawn run
        new_running_list = []
        for cell in self.running_list:
            if cell.label == 'mother':
                new_running_list.append(cell)
            elif cell.label == 'daughter' and cell.age < 3:
                new_running_list.append(cell)
            else:
                pass

        if len(new_running_list) > self.max_pop:
            new_running_list = new_running_list[0:self.max_pop]
        
        self.running_list = new_running_list
        
    def write_df(self):
        # write population history to df
        data = self.get_current_stats()
        self.history.loc[len(self.history)] = data
    
    ### Lump function
        
    def full_ts_pop(self,t):
        self.write_df() # write history
        self.run_trans_counter() # who transitions
        if t%4 == 0:
            self.run_spawn() # what is made
        self.cull() # selection due to capping


    
    ### Helper/debug functions

    def get_current_stats(self):
        # just gathers info on pop breakdown, called by write_df
        yellow_count = 0
        blue_count = 0
        daughter_count = 0
        for part in self.running_list:
            if part.state_circ == 'yellow':
                yellow_count+=1
            elif part.state_circ == 'blue':
                blue_count+=1
            if part.label == 'daughter':
                daughter_count+=1
        return [yellow_count,blue_count,daughter_count]
          
    def print_all_vars(self):
        # prints cell object attributes, for debugging
        for part in self.running_list:
                print(vars(part))



def run_sim(pop_seed,max_pop,n_divs,trans_kwargs,spawn_kwargs):
    '''Function that executes a defined simulation
    inputs:
    pop_seed: [int,int], yellow,blue initial pop size
    max_pop: int, max population size to simulate (used in capping)
    n_divs: int number of divisons to simulate
    trans_kwargs: list of form [p_ytob,p_btoy]
    spawn_kwargs: list of form [div_constant,]
    
    output:
    cell_env object, really only care about cell_env.history
    '''
    CV = cell_env(max_pop=max_pop)
    CV.meta = [pop_seed,max_pop,n_divs,trans_kwargs,spawn_kwargs]
    CV.initialize_cells_det(pop_seed=pop_seed,trans_kwargs=trans_kwargs,spawn_kwargs=spawn_kwargs)
    for n in tqdm(range(0,n_divs)):
        CV.full_ts_pop(t=n+1)
          
    return CV








