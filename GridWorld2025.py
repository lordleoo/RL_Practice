
get_ipython().magic('%reset -f')
get_ipython().magic('%cls')

# from selenium.webdriver.chrome.options import Options
# from selenium import webdriver
import pandas as pd
import numpy  as np
import os, sys, copy

# execfile(r'C:\Users\mohandes\Documents\MATLAB\startup.py')
execfile(r"C:\Users\Lord_\Documents\Python\startup.py")
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

#%% Class Definition
class Grid:
    def __init__(self, *,
                 Map,
                 V_init     = 0,
                 Wall_symb  = '#',
                 Actions    = {'L'      : [ 0,-1],
                               'D'      : [ 1, 0],
                               'R'      : [ 0, 1],
                               'U'      : [-1, 0],

                               # 'UL'     : [-1,-1], #diagonal moves: Up left
                               # 'DL'     : [ 1,-1],
                               # 'UR'     : [-1, 1],
                               # 'DR'     : [ 1, 1],

                               # '2L'     : [ 0,-2], #leap left
                               # '2D'     : [ 2, 0], #leap down
                               # '2R'     : [ 0, 2], #lead right
                               # '2U'     : [-2, 0], #leap up

                               # 'K1'     : [ 1,-2], #knight move
                               # 'K2'     : [ 2,-1],
                               # 'K3'     : [ 2, 1],
                               # 'K4'     : [ 1, 2],
                               # 'K5'     : [-1, 2],
                               # 'K6'     : [-2, 1],
                               # 'K7'     : [-2,-1],
                               # 'K8'     : [-1,-2],
                               },
                 Trans = 1, #FIXME# Transition matrix. not implemented.
                 Reward = 0, #FIXME# Reward/penalty for every step you make. not implemented
                 **kwargs):

        if isinstance(Map,np.ndarray): # if the map is a numpy array, take it as is
            self._Grid = Map;
        elif isinstance(Map,pd.DataFrame): # if the map is a panda DF, convert it to a numpy array
            self._Grid = Map.astype(object).values

        elif isinstance(Map,dict) and 'FileName' in Map: # if the map is a dictionary, then:
            FileName        = Map['FileName']
            xls_kwargs      = {**Map}
            xls_kwargs.pop('FileName');
            FullSheet       = pd.read_excel(FileName, header = None, **xls_kwargs)
            self._Grid      = FullSheet.astype(object).values

        self._Walls         = (self._Grid==Wall_symb) #This maps walls only

        if not np.all([*self._Walls[:,0],*self._Walls[:,-1],*self._Walls[0,:],*self._Walls[-1,:]]):
            self._Grid      = np.pad(self._Grid,(1,1),constant_values=(Wall_symb));
            self._Walls     = (self._Grid==Wall_symb) #This maps walls only

        self._Dims          = np.shape(self._Grid);
        Which_R             = np.vectorize(lambda x: (type(x) in [float, int]) and (not np.isnan(x)) )(self._Grid)
        # self._Which_R       = Which_R;
        self._R             = np.zeros_like(self._Grid); #Can the reward depend on where you came from, for the same destination?
        self._R[Which_R]    = self._Grid[Which_R]
        self._Trmnl         = self._R.astype(bool)

        self._Vs            = np.ones_like(self._R) * V_init; #creates a deep copy; np.asarray(...) does NOT work. np.asarray creates a shallow copy
        self._Vs0           = self._Vs.astype(float);
        self._Qsa           = np.ones([*self._Grid.shape,len(Actions)]).astype(float) * V_init;


        self._Actions       = Actions;
        self._AAbbrv        = [i[:3] for i in list(self._Actions.keys())]


        self._A             = np.ones([*self._Grid.shape,len(Actions)]).astype(bool);
        NoUp    = np.roll(self._Walls, axis=0, shift= 1);    NoUp[ 0,:] = False; self._A[:,:,-1][NoUp]    = False; # shift down  by one
        NoDown  = np.roll(self._Walls, axis=0, shift=-1);  NoDown[-1,:] = False; self._A[:,:, 1][NoDown]  = False; # shift up    by one
        NoLeft  = np.roll(self._Walls, axis=1, shift= 1);  NoLeft[:, 0] = False; self._A[:,:, 0][NoLeft]  = False; # shift right by one
        NoRight = np.roll(self._Walls, axis=1, shift=-1); NoRight[:,-1] = False; self._A[:,:, 2][NoRight] = False; # shift left  by one; Truncate last column
        self._A[ :,  0,  0]     = False; # on the LeftMost   Column self._A[ :, 0], moving left  is not an option
        self._A[ :, -1,  2]     = False; # on the RightMost  Column self._A[ :,-1], moving right is not an option
        self._A[ 0,  :, -1]     = False; # on the TopMost    row    self._A[ 0, :], moving up    is not an option
        self._A[-1,  :,  1]     = False; # on the LowerMost  row    self._A[-1, :], moving down  is not an option
        self._A[self._Walls,:]  = False; # If you are a wall, you can't move
        self._A[Which_R,:]      = False; # Rewarded squares are terminal states; no actions from these;
        self._Pi = self._A.astype(float); #i dont want to set the policy_weight on terminal squares to zeros.

        self._Start             = np.vectorize(lambda x: x=='S')(self._Grid);
        self._Position          = np.argwhere(self._Start).ravel()

        #######################################################################################
        self._GridMap   = self._Grid.astype('<U36')
        SquareSize      = np.max(np.vectorize(len)(self._GridMap))
        SquareSize      = int(np.ceil(SquareSize/2)*2); #make it even
        self._SquareSize= SquareSize
        self._GridMap[self._Grid=='#']=np.str_(u'\u2588').astype('<U30'); #Walls
        self._GridMap[self._Grid=='S']=np.str_(u'\u2605').astype('<U30'); #Starting point
        self._PositionMark = [np.str_(i).astype('<U30').center(SquareSize) for i in [u'\u2662',u'\u2666']];
        if len(self._Position)>0: self._GridMap[tuple(self._Position)]=self._PositionMark[-1]
        RawRuler = [i for i in range(max(self._Dims))]
        RawRuler[ 0]    = ''
        RawRuler[-1]    = ''
        RawRuler        = [str(i).center(SquareSize) for i in RawRuler]
        Ruler           = np.vectorize(lambda x: ColorStr(x,bg='K',fg='w'))(RawRuler)

        self._GridMap[ :, 0] = Ruler[:self._GridMap.shape[0]]
        self._GridMap[ :,-1] = Ruler[:self._GridMap.shape[0]]
        self._GridMap[ 0, :] = Ruler[:self._GridMap.shape[1]]
        self._GridMap[-1, :] = Ruler[:self._GridMap.shape[1]]
        self._GridMap[ 0, 0] = Ruler[0]
        self._GridMap[ 0,-1] = Ruler[0]
        self._GridMap[-1, 0] = Ruler[0]
        self._GridMap[-1,-1] = Ruler[0]

        self._GridMap        = np.vectorize(lambda x: x.center(SquareSize, u'\u2588' if x==u'\u2588' else ' '))(self._GridMap)

        self._GridMap[self._R<0]   = np.vectorize(lambda x: ColorStr(x.center(SquareSize), bg='r'))(self._GridMap[self._R<0])
        self._GridMap[self._R>0]   = np.vectorize(lambda x: ColorStr(x.center(SquareSize), bg='g', fg='k'))(self._GridMap[self._R>0])
        if np.sum(self._Start)>0: self._GridMap[self._Start] = np.vectorize(lambda x: ColorStr(x.center(SquareSize), bg='y', fg='k'))(self._GridMap[self._Start])

        self._GridMapRaw           = copy.deepcopy(self._GridMap);
        self._GridMap[np.char.strip(self._GridMap)=='nan'] = ''.center(SquareSize)

        [print(*i,sep='') for i in self._GridMap]


#%% Properties
    @property
    def GridMap(self):
        [print(*i,sep='') for i in self._GridMap];

    @property
    def Walls(self):
        return self._Walls;

    @property
    def Paths(self):
        return (self._Walls.__or__(self._R.astype(bool))).__invert__();

    @property
    def Rewards(self):
        return self._R

    # @property
    # def ActionSpace(self):
    #     return self._A

    @property
    def TerminalStates(self):
        return self._Trmnl

    @property
    def StateValues(self):
        return self._Vs0

    @property
    def StateActionValues(self):
        return self._Qsa

    @property
    def BestAction(self):
        return np.argmax(self.Q,axis=-1)

#%% V and Q ##############################
    @property
    def V(self):
         return self._Vs0

    @V.setter
    def V(self, NewValue):
            self._Vs = NewValue

###########################################
    @property
    def Q(self):
         return self._Qsa

    @Q.setter
    def Q(self, NewValue):
            self._Qsa = NewValue

###########################################
    @property
    def Pi(self):
         return self._Pi

    @Pi.setter
    def Pi(self, NewValue):
            self._Pi = NewValue

#%% Move ############################################################
    def Move(self, Actn,
                   pos     = None,
                   serious = True):

        if pos is None:
            pos = self._Position
        else:
            self._Position = pos;


        if self._Trmnl[tuple(pos)]: #is True, terminal
            if serious:
                print('Game terminated at square: [%i, %i] with Reward: %f' %(*pos, self._R[tuple(pos)]))
                self._Position = None;
            return None, \
                   self._R[tuple(pos)]


        shift       = self._Actions[Actn]
        valid_move  = [False]*2;
        sup_pos     = [*pos]

        # TODO incorporate transition probability;
        for i in [0,1]:
            sup_pos[i]   += shift[i]
            valid_move[i] = True if (sup_pos[i] in range(1,self._Dims[i])) else False;

        if all(valid_move) and (not self._Walls[tuple(sup_pos)]):
            if serious:
                # self._GridMap[tuple(self._Position)]    = self._GridMap[tuple(self._Position)].replace(self._PositionMark[-1],self._PositionMark[0]);
                self._Position                          = sup_pos
                # self._GridMap[tuple(self._Position)]    = self._PositionMark[-1]

        else: #not a valid move
            sup_pos     = [*pos]

        return  sup_pos, \
                self._R[tuple(sup_pos)]

        # try:
        #     print(self._R[tuple(self._Position)])
        # except Exception as RErr:
        #     pass


#####################################################################
    def RoadPlan(self):
        self._GridMapRaw[self.Paths]
        with np.nditer(self._GridMapRaw, flags=['multi_index','refs_ok']) as it:
            ############# Iterating over all states (squares in the grid) #############
            for _ in it:
                idx = it.multi_index

                if self.Paths[tuple(idx)]:
                    self._GridMapRaw[tuple(idx)] = self._AAbbrv[self.BestAction[tuple(idx)]].center(self._SquareSize)
        return print('\n'.join([''.join(i) for i in self._GridMapRaw]))


#####################################################################

    def __repr__(self): # self;
        # [print(*i,sep='') for i in self._GridMap];
        return '\n'.join([''.join(i) for i in self._GridMap])

    def __str__(self): # print(self)
        # [print(*i,sep='') for i in self._GridMap];
        return self.__repr__()

#####################################################################

#%% Import Grid as excel
InputData0=  \
[{
  'FileName'     : r"E:\LIST\Produce\Python practice\RL\GridWorld.xlsx",
  'Range'        : 'A:D',
    'Sheet'        : 'Sheet1',
    # 'Sheet'        : 'Sheet2',
    # 'Sheet'        : 'Sheet6',
},
]
V_init = 0;

InputData = InputData0[0]

excelName, Range, Sheet = InputData.values();

print('Ignore the warning about \"conditional formatting\" below')
FullSheet               = pd.read_excel(excelName, header = None, sheet_name = Sheet) # Conditional formatting warning here
FullSheet               = FullSheet.astype(object).values

Actions = { 'L'     : [ 0,-1],
            'D'     : [ 1, 0],
            'R'     : [ 0, 1],
            'U'     : [-1, 0],
            # 'UL'    : [-1,-1], #diagonal moves: Up left
            # 'DL'    : [ 1,-1],
            # 'UR'    : [-1, 1],
            # 'DR'    : [ 1, 1],

            # '2L'    : [ 0,-2], #leap
            # '2D'    : [ 2, 0],
            # '2R'    : [ 0, 2],
            # '2U'    : [-2, 0],

            # '2UL'   : [-2,-2], #leap diagonally
            # '2UR'   : [-2, 2],
            # '2DL'   : [ 2,-2],
            # '2DR'   : [ 2, 2],

            # 'K1'    : [ 1,-2], #knight move
            # 'K2'    : [ 2,-1],
            # 'K3'    : [ 2, 1],
            # 'K4'    : [ 1, 2],
            # 'K5'    : [-1, 2],
            # 'K6'    : [-2, 1],
            # 'K7'    : [-2,-1],
            # 'K8'    : [-1,-2],
           }

Act_Probs = dict.fromkeys(Actions.keys(), 1/len(Actions));


MyGrid = Grid(Map = FullSheet, Actions=Actions)
MyGrid.Walls
MyGrid.Paths

#%%

delta_target    = [1e-5, 5e-2]
iter_max        = [3, 2e1]; # 1 corresponds to value iteraiton, np.inf corresponds to policy iteration;
discount_factor = 1;
itr             = [0 , 0]
delta           = [1e6, 1e6]
epsilon_greed   = lambda itr, eps_range, iter_max=iter_max[1]: np.exp(-7*itr / iter_max)*(np.ptp(eps_range)) + min(eps_range) if np.ptp(eps_range)>0 else np.max(eps_range);
eps_range       = [0, 0.8];

# sys.exit()
while (itr[1] < iter_max[1]) and (delta[1] > delta_target[1]):
    delta   = [1e6, 1e6]
    itr[1] += 1
    itr[0]  = 0
    # print('Starting outer iter#%i' %itr[0])

    ###################### EVALUATE POLICY ######################
    # Iterate till convergence
    Actions_set  = list(Actions.keys());
    Actions_idx     = {i: np.argwhere(np.asarray(Actions_set) == i).ravel() for i in Actions}
    Transition_prob = np.ones_like(Actions_set).astype(int);
    while (itr[0] < iter_max[0]) and (delta[0] > delta_target[0]):
        # print('Iteration #%i' %itr[1])
        Vs0 = copy.deepcopy(MyGrid.V);

        # Go through all states
        with np.nditer(MyGrid._Grid, flags=['multi_index','refs_ok']) as it:
            ############# Iterating over all states (squares in the grid) #############
            for _ in it:
                idx = it.multi_index
                # print('Doing state <%2i, %2i>' %(idx))
                if MyGrid.Walls[tuple(idx)]:
                    # unless you're humpty-dumpy, then skip this cell. if you're humpty-dumpty, get life insurance.
                    continue;
                elif MyGrid._Trmnl[tuple(idx)]:
                    MyGrid.V[tuple(idx)] = MyGrid._R[tuple(idx)];
                    continue;
                # wrong. don't activate. elif (itr[0] > 5) and (abs(MyGrid.V[tuple(idx)])<1): # this cell is probably unreachable. skip it.
                #     continue;
                else:
                    #theoretically, i should have a transition probability tensor with dimension
                    # states x actions x states. this would get huge, and we dont do it in practice.
                    # in fact, even if we had the environment's model, modeling this would get so
                    # expensive (storage wise), that eventhough we have the environment model
                    # we resort to approximation functions.


                    Expeditions     = np.asarray([MyGrid.Move(x, pos=idx, serious=False) for x in Actions_set],dtype=object)
                    Rewards         = Expeditions[:,-1]
                    Vs0_next        = np.vectorize(lambda x: 0 if x==None else Vs0[tuple(x)])(Expeditions[:,0]); # we V(s'), s' is the next state, not the current. i made a mistake here the first time i coded this
                    UpdateSet       = np.asarray([r + discount_factor * v for r,v in zip(Rewards, Vs0_next)]); #i made a mistake here the first time i did this. i used: discount_factor * reward + V_old(s); V_old(s) is wrong. we need gamma * V_old(s')

                    UpdateDict      = {i: np.average(UpdateSet[j], weights = Transition_prob[j]) \
                                       for i,j in Actions_idx.items()} # expectation  for each action. this NOT up to you. you can NOT take max here, because you can't control transition probability;

                    MyGrid.V[tuple(idx)]  = np.average(list(UpdateDict.values()),
                                                       weights = tuple(MyGrid._Pi[tuple(idx)]));

                    MyGrid.Q[tuple(idx)]  = list(UpdateDict.values())

                    # actual_update will reflect my policy.
                    # whether i take blind average or greedy or epsilon-greedy, is represented by Policy_weights

                    Rewards *= 0 #just to reset

        delta[0] = np.max(abs(MyGrid.V - Vs0))
        if (itr[0] in np.ceil(np.arange(0, 1.1, 0.05) * iter_max[0]) or delta[0] < delta_target[0]):
            print('Inner Iter#%3i, MAX value  update size = %g' %(itr[0], delta[0]))
        itr[0] += 1;

    # outside the while loop
    #############################################################


    ###################### IMPROVE POLICY ######################
    # Go through all states. (No need to repeat this. we're not seeking a convergence)

    Pi0 = copy.deepcopy(MyGrid.Pi);
    epsilon = epsilon_greed(itr[1], eps_range = eps_range)
    with np.nditer(MyGrid._Grid, flags=['multi_index','refs_ok']) as it:
        ############# Iterating over all states (squares in the grid) #############
        for _ in it:
            idx = it.multi_index
            # print('Doing state <%2i, %2i>' %(idx))
            if MyGrid.Walls[tuple(idx)]:
                # are you humpty dumpty? if not, then continue.
                continue;
            elif MyGrid._Trmnl[tuple(idx)]:
                MyGrid.V[tuple(idx)] = MyGrid._R[tuple(idx)];
                continue;
            else:
                    Pi_new  = np.ones_like(MyGrid.Q[tuple(idx)]) * (epsilon/len(MyGrid._Actions.keys()))
                    Pi_new[MyGrid.BestAction[tuple(idx)]] += (1-epsilon)
                    MyGrid.Pi[tuple(idx)] = Pi_new;
                    del Pi_new;
                    ########### Implement the policy ###########



    delta[1] = np.sum(abs(MyGrid.Pi - Pi0))
    print('Outer iter#%3i, SUM policy update size = %g. ' %(itr[1], delta[1]))

                    # perhaps try learning rate here.

    ###################### IMPROVE POLICY ######################

MyPrint = lambda *args: print(*args, sep='\n',end='\n'*2)
MyPrint('Grid',MyGrid);
MyPrint('Values',MyGrid.V);
MyPrint('Best Action',MyGrid.BestAction);
MyPrint('Road Plan',MyGrid.RoadPlan());
sys.exit()

#%% play
get_ipython().magic('%cls')

ValidStartingPoints = np.argwhere(MyGrid.Paths);
NextState = ValidStartingPoints[np.random.choice(len(ValidStartingPoints))]
learning_rate   = 1e-3;
iter_test = 0;
History = [];

print('Starting from point %s' %str(NextState))
while NextState is not None:
    iter_test+=1;
    CurrentState = NextState;
    if np.random.rand() < epsilon_greed(iter_test, eps_range=eps_range): #take a random action
        TheAction = np.random.choice(list(MyGrid._Actions.keys()))
        StepType = 'Random'

    else: #take max
        TheAction = list(MyGrid._Actions.keys())[MyGrid.BestAction[tuple(CurrentState)]];
        StepType = 'Greedy'

    NextState, Reward = MyGrid.Move(TheAction,
                                    pos      = CurrentState,
                                    serious  = True);

    print('%6s Step %i: Moved %2s: %8s >> %8s' %(StepType, iter_test, TheAction, CurrentState, NextState));

print(MyGrid)



#%% Monte-Carlo

# Actions_set   = np.random.choice(list(Actions.keys()), p=list(Act_Probs.values()), size=N_rand_acts)
# Transition_prob  = np.ones_like(Actions_set).astype(int); #i wouldn't know transition. transition would reveal itself in the sample


ChainLength = int(2e3);
N_chains    = int(1e3)
RewMat      = [];
ValidStartingPoints = np.argwhere(MyGrid.Paths);
MCChains = np.random.choice(list(Actions.keys()), size=[ChainLength,N_chains])

for iter_chain, chain in enumerate(MCChains):
    # iter_chain = 0; chain = MCChains[iter_chain]
    StartPoint = ValidStartingPoints[np.random.choice(len(ValidStartingPoints))]
    NextState  = StartPoint;
    Reward = 0;
    for iter_step, step in enumerate(chain):
        # iter_step = 0; step = chain[iter_step]
        NextState, rew = MyGrid.Move(step, pos = NextState);
        Reward += rew;
        if NextState is None:
            break;
    RewMat.append([StartPoint.tolist(), Reward]);
