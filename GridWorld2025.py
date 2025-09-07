
# get_ipython().magic('%reset -f')
# get_ipython().magic('%cls')

%cls
%reset -f

# from selenium.webdriver.chrome.options import Options
# from selenium import webdriver
import pandas as pd
import numpy  as np
import os, sys, copy
import FileUtilities as FU

from multiprocessing import Pool, cpu_count
n_cores = cpu_count()
# execfile(r'C:\Users\mohandes\Documents\MATLAB\startup.py')
execfile(r"C:\Users\Lord_\Documents\Python\startup.py")
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)

dict_findValue = lambda TheDict, val: [key for key,value in TheDict.items() if value==val]

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
                 invalidPenalty = 0, # penalty for taking an invalid move
                 terminal = None,
                 # Trans = 1, #FIXME# Transition matrix. not implemented.
                 # Reward = 0, #FIXME# Reward/penalty for every step you make. not implemented
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
        self._Dims0         = np.shape(self._Grid); # Grid shape before padding

        if not np.all([*self._Walls[:,0],*self._Walls[:,-1],*self._Walls[0,:],*self._Walls[-1,:]]):
            self._Grid      = np.pad(self._Grid,(1,1),constant_values=(Wall_symb));
            self._Walls     = (self._Grid==Wall_symb) #This maps walls only
        self._NotWalls      = (self._Grid!=Wall_symb) #This maps NONE WALLS only (internal walls of a maze are also labelled walls)

        self._Dims          = np.shape(self._Grid);
        Which_R             = np.vectorize(lambda x: (type(x) in [float, int]) and (not np.isnan(x)) )(self._Grid)
        # self._Which_R       = Which_R;
        self._R             = np.zeros_like(self._Grid); #Can the reward depend on where you came from, for the same destination?
        self._R[Which_R]    = self._Grid[Which_R]
        self._invalidPenalty = invalidPenalty; # penalty for taking an invalid move
        self._Jackpot       = np.max(self._R)


        # Terminal position --------------------------------------------------------------------------
        if isinstance(terminal,None.__class__) or (isinstance(terminal,str) and (terminal.lower().startswith('max') or terminal.lower().startswith('jackpot'))):
             self._Trmnl         = np.zeros_like(self._Grid,dtype=bool)
             # np.argmax finds only one instance of maximum. i want to find all instances
             # self._Trmnl_idx   = np.unravel_index(np.argmax(self._R, axis=None), self._R.shape)
             Trmnl_LinIdx        = np.flatnonzero(self._R == self._Jackpot)
             self._Trmnl_Idx     = np.vstack(np.unravel_index(Trmnl_LinIdx, self._R.shape)).T
             for  idx in self._Trmnl_Idx:
                  self._Trmnl[tuple(idx.tolist())] = True
             del idx;

        elif isinstance(terminal,str) and terminal.lower().startswith('reward'):
             self._Trmnl         = self._R.astype(bool) # This terminates at any NON-zero reward

        elif isinstance(terminal,(list,np.ndarray)):
             terminal_ = np.array(terminal,dtype=object)
             if   terminal_.shape[1] > 2:
                  if   terminal_.shape == self._Dims:
                       self._Trmnl = terminal;
                  elif terminal_.shape == self._Dims0:
                       self._Trmnl = np.zeros_like(self._Grid);
                       self._Trmnl[1:-1,1:-1] = terminal_.astype(bool)
             else:
                  margin = 1 if self._Dims0!=self._Dims else 0;
                  for pair in terminal_:
                       idxx, idxy = pair
                       self._Trmnl[(idxx+margin,idxy+margin)]  = True

        # Start point --------------------------------------------------------------------------
        self._Start           = np.vectorize(lambda x: x in ['S','s'])(self._Grid);
        Start_LinIdx          = np.flatnonzero(self._Start).T
        self._Start_Idx       = np.vstack(np.unravel_index(Start_LinIdx, self._Start.shape)).T

        self._Position        = np.argwhere(self._Start).ravel()

        # V_init --------------------------------------------------------------------------
        if isinstance(V_init,(int,float,bool)):
             self._Vs       = np.ones_like(self._R) * V_init; #creates a deep copy; np.asarray(...) does NOT work. np.asarray creates a shallow copy
             self._Vs0      = self._Vs.astype(float);
             # self._Qsa      = np.ones([*self._Grid.shape,len(Actions)]).astype(float) * V_init;
             self._Qsa      = np.zeros([*self._Grid.shape,len(Actions)]).astype(float)

        else:
             if   isinstance(V_init,list):
                  V_init = np.array(V_init,dtype=object)
             if   isinstance(V_init,np.ndarray):
                  pass # do nothing. i put this here for completivity
             else:
                  raise TypeError('Not sure what V_init is. is the number of columns in the maze == 2?')

             # breakpoint()
             if   V_init.shape[1] > 2:
                  V_init_valid = np.vectorize(lambda x: (not isinstance(x,str)) and (not np.isnan(x)))(V_init)
                  V_init_idx = np.vstack([np.where(V_init_valid),V_init[V_init_valid]]).T # This is slight faster than the line below
                  # V_init_idx = np.hstack([np.argwhere(V_init_valid),V_init[V_init_valid,None]])
                  margin = 1 if self._Dims0!=self._Dims else 0;

             elif V_init.shape[1] == 2:
                  V_init_idx = np.vstack([np.hstack(xx) for xx in V_init]) # i assume first two will be indices and third will be value

             # self._Vs0[V_init_valid]      = V_init[V_init_valid].astype(float)
             # self._Vs[V_init_valid]       = V_init[V_init_valid].astype(float)

             self._Vs       = np.zeros_like(self._R); #creates a deep copy; np.asarray(...) does NOT work. np.asarray creates a shallow copy
             self._Vs0      = self._Vs.astype(float);

             for  triple in V_init_idx:
                  idxx, idxy , val = triple
                  self._Vs[(idxx+margin,idxy+margin)]  = val
                  self._Vs0[(idxx+margin,idxy+margin)] = val

             # self._Qsa           = np.tile(self._Vs0[:,:,None],(1,1,len(Actions)))/len(Actions)
             # self._Qsa           = np.tile(self._Vs0[:,:,None],(1,1,len(Actions)))
             self._Qsa             = np.zeros((*self._Vs0.shape,len(Actions)))
        self._BestAction      = np.zeros_like(self._Vs,dtype=int)

        ##### Old initializaiton
        # self._Vs            = np.ones_like(self._R) * V_init; #creates a deep copy; np.asarray(...) does NOT work. np.asarray creates a shallow copy
        # self._Vs0           = self._Vs.astype(float);
        # self._Qsa           = np.ones([*self._Grid.shape,len(Actions)]).astype(float) * V_init;

        # Actions --------------------------------------------------------------------------
        # breakpoint()
        self._Actions       = Actions;
        self._Actions_set  = list(Actions.keys())
        self._Actions_Nam2Idx = {key:itr for itr,key in enumerate(Actions.keys())}
        # self._Actions_Nam2Idx = {i: np.argwhere(np.asarray(MyGrid._Actions_set) == i).item() for i in Actions}
        self._Actions_Idx2Nam = {itr:key for itr,key in enumerate(Actions.keys())}
        self._Actions_Mov2IdxNam = {tuple(val):(itr,key) for itr,(key,val) in enumerate(Actions.items())}

        self._A             = np.ones([*self._Grid.shape,len(Actions)]).astype(bool);
        self._A[self._Walls,:]  = False; # If you are a wall, you can't move
        self._Pi            = np.zeros_like(self._A,dtype=float); #i dont want to set the policy_weight on terminal squares to zeros.
        # breakpoint()
        self._StillAction   = self._Actions_Mov2IdxNam.get((0,0))


        # Initialization for policy PI. this is optional to help the algorithm. the algorithm can still update this as needed
        with np.nditer(self._Grid, flags=['multi_index','refs_ok']) as it:
             for _ in it:
                  idx = it.multi_index
                  if   self._Walls[tuple(idx)]:
                       continue

                  for  act_key, act_val in self._Actions.items():
                       _, _, _, _, isValid = self.Move(Actn=act_key, pos=idx, serious=False)
                       self._A[*idx,self._Actions_Nam2Idx[act_key]] = isValid

                  if not isinstance(self._StillAction,None.__class__) and self._Trmnl[tuple(idx)]: # This is optional. make INIITIAL policy on terminal state be: stay still. the algorithm can still update this
                       self._Pi[*idx,:] = 0; # set policy to 0 for all, except the "stay still" action
                       self._Pi[*idx,self._StillAction[0]] = 1; # do this, but Actions for remaining directions are still TRUE if valid

                  else: # If not a terminal state
                       self._Pi[*idx,self._A[*idx,:]] = 1/(self._A[*idx,:].astype(float).sum()); #i dont want to set the policy_weight on terminal squares to zeros.
                  del idx;
        #--------------------------------------------------------------------------
        if kwargs.get('ActionsPrint') == None:
             self._AAbbrv        = [i[:3] for i in list(self._Actions.keys())]
        else:
             ActionsPrint = kwargs.get('ActionsPrint')
             if not isinstance(ActionsPrint,dict):
                  raise TypeError('Provided keyword \"ActionsPrint\" has type %s!. The printing of set of actions should be a dictionary.' %type(ActionsPrint))
             self._AAbbrv        = [ActionsPrint[xx] for xx in self._Actions.keys()] # This throws an error if the key does not exist in ActionsPrint
             # self._AAbbrv        = [ActionsPrint.get(xx) for xx in self._Actions.keys()] # This doesn't throw an error and I want the error

        # I deactivated this because it is written such that action 0 is left, action 1 is down, etc....
        # NoUp    = np.roll(self._Walls, axis=0, shift= 1);    NoUp[ 0,:] = False; self._A[:,:,-1][NoUp]    = False; # shift down  by one
        # NoDown  = np.roll(self._Walls, axis=0, shift=-1);  NoDown[-1,:] = False; self._A[:,:, 1][NoDown]  = False; # shift up    by one
        # NoLeft  = np.roll(self._Walls, axis=1, shift= 1);  NoLeft[:, 0] = False; self._A[:,:, 0][NoLeft]  = False; # shift right by one
        # NoRight = np.roll(self._Walls, axis=1, shift=-1); NoRight[:,-1] = False; self._A[:,:, 2][NoRight] = False; # shift left  by one; Truncate last column
        # self._A[ :,  0,  0]     = False; # on the LeftMost   Column self._A[ :, 0], moving left  is not an option
        # self._A[ :, -1,  2]     = False; # on the RightMost  Column self._A[ :,-1], moving right is not an option
        # self._A[ 0,  :, -1]     = False; # on the TopMost    row    self._A[ 0, :], moving up    is not an option
        # self._A[-1,  :,  1]     = False; # on the LowerMost  row    self._A[-1, :], moving down  is not an option
        # self._A[Which_R,:]      = False; # Rewarded squares are terminal states; no actions from these;

        # for idx in self._Trmnl_Idx: # Deactivated this because algorithm should automatically find out that "stay" is the best action
        #      self._A[*tuple(idx),:]= False; # Only terminal square has no action

        # breakpoint()

        #######################################################################################
        self._GridMap   = self._Grid.astype('<U36')
        # SquareSize      = np.char.str_len(self._GridMap).max(axis=None).item() # Faster
        SquareSize      = np.max(np.vectorize(len)(self._GridMap))
        SquareSize      = int(np.ceil(SquareSize/2)*2); #make it even
        self._SquareSize= SquareSize
        self._GridMap[self._Grid=='#']=np.str_(u'\u2588').astype('<U30'); #Walls
        # self._GridMap[self._Grid=='S']=np.str_(u'\u2605').astype('<U30'); #Starting point # ord('★') = 9733
        self._GridMap[self._Grid=='S']=np.str_(u'\u2666').astype('<U30'); #Starting point # ord('★') = 9733
        # self._PositionMark = [np.str_(i).astype('<U30').center(SquareSize) for i in [u'\u2605',u'\u2662',u'\u2666']];
        # if len(self._Position)>0: self._GridMap[tuple(self._Position)]=self._PositionMark[-1]
        RawRuler = [i for i in range(max(self._Dims))]
        RawRuler[ 0]    = ''
        RawRuler[-1]    = ''
        RawRuler        = [str(i).center(SquareSize) for i in RawRuler]
        Ruler           = np.vectorize(lambda x: ColorStr(x,bg='K',fg='w'))(RawRuler) # ColorStr is from startup.py; it enters the ugly text to create colored blocks
        self._BWRoadsMap = copy.deepcopy(self._GridMap) # Without the color string
        self._BWRoadsMap[ :, 0] = Ruler[:self._BWRoadsMap.shape[0]]
        self._BWRoadsMap[ :,-1] = RawRuler[:self._BWRoadsMap.shape[0]]
        self._BWRoadsMap[ 0, :] = RawRuler[:self._BWRoadsMap.shape[1]]
        self._BWRoadsMap[-1, :] = RawRuler[:self._BWRoadsMap.shape[1]]
        self._BWRoadsMap[ 0, 0] = RawRuler[0]
        self._BWRoadsMap[ 0,-1] = RawRuler[0]
        self._BWRoadsMap[-1, 0] = RawRuler[0]
        self._BWRoadsMap[-1,-1] = RawRuler[0]

        self._GridMap[ :, 0] = Ruler[:self._GridMap.shape[0]]
        self._GridMap[ :,-1] = Ruler[:self._GridMap.shape[0]]
        self._GridMap[ 0, :] = Ruler[:self._GridMap.shape[1]]
        self._GridMap[-1, :] = Ruler[:self._GridMap.shape[1]]
        self._GridMap[ 0, 0] = Ruler[0]
        self._GridMap[ 0,-1] = Ruler[0]
        self._GridMap[-1, 0] = Ruler[0]
        self._GridMap[-1,-1] = Ruler[0]

        # print(*self._GridMap)

        self._GridMap        = np.vectorize(lambda x: x.center(SquareSize, u'\u2588' if x==u'\u2588' else ' '))(self._GridMap)

        # self._GridMap[self._R<0]   = np.vectorize(lambda x: ColorStr(x.center(SquareSize), bg='r'))(self._GridMap[self._R<0])
        # self._GridMap[self._R>0]   = np.vectorize(lambda x: ColorStr(x.center(SquareSize), bg='g', fg='k'))(self._GridMap[self._R>0])
        # tic=time.time(); print(time.time() - tic)
        # looping once over all rewards is faster than Vectorize twice.
        # Vectorize fails if there are no negative blocks, or no positive blocks
        R_idx = np.argwhere(self._R.astype(bool))
        for r_idx in R_idx:
             r = self._R[tuple(r_idx)]
             self._GridMap[tuple(r_idx)] = ColorStr(str(r).center(SquareSize), bg='g' if r>0 else 'r', fg='k')

        if np.sum(self._Start)>0:
             self._GridMap[self._Start] = np.vectorize(lambda x: ColorStr(x.center(SquareSize), bg='y', fg='k'))(self._GridMap[self._Start])

        # self._GridMapRaw           = copy.deepcopy(self._GridMap);
        which_nan = (self._Grid!=self._Grid) # nan is unique in the sense it doesnt equal itself
        self._GridMap[which_nan] = ''.center(SquareSize) # faster than the one below; and more robust
        # self._GridMap[np.char.strip(self._GridMap)=='nan'] = ''.center(SquareSize)

        # [print(*i,sep='') for i in self._GridMap]

        # print('\n'.join([''.join(xx) for xx in self._GridMap]));
        # breakpoint()

#%% Properties
    def Bool2Idx(self,BoolMap):
        indcs = np.argwhere(BoolMap) # This sweeps horizontally
        indcs2 = indcs[np.lexsort((indcs[:,0],indcs[:,1]))] # This way, it sweeps vertically, same way as VECTORIZE does (for consistency with old code only)
        out=[tuple(x.tolist()) for x in indcs2];
        return out;

    @property
    def GridMap(self):
        [print(*i,sep='') for i in self._GridMap];

    @property
    def Walls(self):
        return self._Walls;

    @property
    def Paths(self):
        # return (self._Walls.__or__(self._R.astype(bool))).__invert__(); # This declares any box with a reward as NOT a path
        # return (self._Walls.__or__(self._Trmnl.astype(bool))).__invert__(); # Only terminal state and walls are NOT paths
        return (self._Walls).__invert__(); # Only terminal state and walls are NOT paths

    @property
    def Paths_idx(self):
        return self.Bool2Idx(self.Paths); # Only terminal state and walls are NOT paths

    @property
    def Clear_Paths(self):
        return self._Walls.__or__(self._R.astype(bool)).__invert__();

    @property
    def Clear_Paths_idx(self):
        return self.Bool2Idx(self.Clear_Paths);

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
    def StateValues0(self):
        return self._Vs0

    @property
    def StateValues(self):
        return self._Vs

    @property
    def StateActionValues(self):
        return self._Qsa

    def UpdateBestAction(self,idx=None):
        if isinstance(idx,None.__class__) or (isinstance(idx,str) and idx.strip().lower()=='all'):
             indcs = self.Paths_idx

        elif isinstance(idx,(list,np.ndarray,tuple)):
             # indcs = np.array(idx,dtype=tuple)
             # indcs = list(map(tuple,idx))
             indcs = [idx]

        for itr_idx, idxidx in enumerate(indcs): # itr_idx=0; idx=indcs[itr_idx]
            if   np.isnan(self.Q[idxidx]).all():
                 beep(F=7000,t=1000) #t in ms
                 breakpoint()
                 raise Exception('Q(%i,%i) is all NaN. BestAction_idx is empty' %(idxidx))

            BestAction_idx  = np.argwhere(self.Q[idxidx] == np.nanmax(self.Q[idxidx]).item()).ravel() # Which actions have best value
            self._BestAction[tuple(idxidx)] = np.random.choice(BestAction_idx,1).item() # if several actions lead to the optimum, choose at random

        return self._BestAction

    @property
    def  BestAction(self):
         return self._BestAction
         # return np.argmax(self.Q,axis=-1)

    @property
    def  image(self):
         rgba = np.ones([*self._Grid.shape,4]).astype(float) # np.ones sets everything to white + transparent
         # rgba[:,:,-1] = 1 # disable transparency (alpha). set opacity to 100%
         rgba[self.Walls,:3] = 0,0,0 # set walls to black

         rewards_norm = plt.Normalize(0,self._R.max())(self._R * (self._R > 0)).astype(float)
         rgba[self._R > 0] = plt.cm.RdYlGn(rewards_norm)[self._R>0]

         penalties_norm = plt.Normalize(self._R.min(),0)(self._R * (self._R < 0)).astype(float)
         rgba[self._R < 0] = plt.cm.RdYlGn(rewards_norm)[self._R<0]

         return rgba
         # plt.imshow(rgba,interpolation='nearest',aspect='auto');

    @property
    def V_df(self):
         V_df = pd.DataFrame(self.V).iloc[1:-1,1:-1].map('{:,.2f}'.format)
         # V_df = pd.DataFrame(self.V).iloc[1:-1,1:-1].map('{:,4.2g}'.format)
         return V_df

#%% V and Q ##############################
    @property
    def size(self):
        return np.prod(self._Dims).item()

###########################################
    @property
    def shape(self):
        return self._Dims

###########################################
    @property
    def V0(self):
        return self._Vs0

###########################################
    @property
    def V(self):
         return self._Vs # This is not a mistake. Vs0 is true # Changed this to Vs to check. return if code doesnt work

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
                   serious = True, # serious about updating current position, or just exploring options
                   ):

        if pos is None:
            pos = self._Position
        else:
            self._Position = pos;


        if self._Trmnl[tuple(pos)]: #is True, terminal
            if serious:
                print('Game terminated at square: [%i, %i] with Reward: %f' %(*pos, self._R[tuple(pos)]))
                self._Position = None;

        shift       = self._Actions[Actn]
        valid_move  = [False]*2;
        sup_pos     = [*pos]

        # TODO incorporate transition probability;
        for i in [0,1]:
            sup_pos[i]   += shift[i]
            valid_move[i] = True if (sup_pos[i] in range(1,self._Dims[i])) else False;

        ValidMove2 = all(valid_move) and (not self._Walls[tuple(sup_pos)])

        if ValidMove2:
             reward = self._R[tuple(sup_pos)]
             next_value = self.V[tuple(sup_pos)]
             if serious:
                  # self._GridMap[tuple(self._Position)]    = self._GridMap[tuple(self._Position)].replace(self._PositionMark[-1],self._PositionMark[0]);
                  self._Position = sup_pos
                  # self._GridMap[tuple(self._Position)]    = self._PositionMark[-1]
        else:
             reward = self._invalidPenalty
             sup_pos     = [*pos]
             next_value = np.nan #FIXME consider feeding the state value of same state
             # next_value = self.V[tuple(sup_pos)] #FIXME consider feeding the state value of same state

        if ValidMove2 and np.isnan(reward):
             breakpoint()
             print('Best action @(%2i,%2i) is [%i] = [%s] with value Q=%g ' %(*pos,
                                                                            self.BestAction[tuple(pos)],
                                                                            self._Actions_set[self.BestAction[tuple(pos)]],
                                                                            self.Q[tuple([*pos,self.BestAction[tuple(pos)]])] ))

             raise Exception('Somehow, the move is valid but reward is NaN')

        return  Actn, \
                (None if serious else sup_pos), \
                reward, \
                float(next_value), \
                ValidMove2

        # try:
        #     print(self._R[tuple(self._Position)])
        # except Exception as RErr:
        #     pass

#####################################################################
    def RoadsMap(self, do_print=True):
         self._RoadsMap   = copy.deepcopy(self._GridMap)
         self._BWRoadsMap = copy.deepcopy(self._Grid)
         which_nan = (self._BWRoadsMap!=self._BWRoadsMap) # nan != nan
         self._BWRoadsMap[which_nan] = ''
         self._BWRoadsMap[MyGrid.Walls] = u'\u2588'

         with np.nditer(self._RoadsMap, flags=['multi_index','refs_ok']) as it:
              ############# Iterating over all states (squares in the grid) #############
              self.UpdateBestAction();
              for _ in it:
                   idx = it.multi_index
                   # breakpoint()
                   # if   self.Paths[tuple(idx)]:
                   if   not self.Walls[tuple(idx)]:
                        if MyGrid._Start[tuple(idx)]:
                             # self._RoadsMap[tuple(idx)]=ColorStr(u'\u2605'.center(self._SquareSize),bg='y',fg='r'); #Starting point
                             # self._BWRoadsMap[tuple(idx)] =u'\u2605'.center(self._SquareSize)
                             self._RoadsMap[tuple(idx)]=ColorStr(chr(9830).center(self._SquareSize),bg='y',fg='r'); #Starting point. ord('♦')=9830
                             self._BWRoadsMap[tuple(idx)] =chr(9830).center(self._SquareSize*0)
                             # the star symbol: "u'\u2605'" is wider than a standard character and messes up alignment

                        elif self._R.astype(bool)[tuple(idx)]:
                             # if this box happens to have some reward, report the action but preserve the color
                             # r_char_len = len(self._R.astype(str)[tuple(idx)]) # this neglects the + sign of a reward
                             rr = self._R[tuple(idx)]
                             r_str = '%+i' %rr
                             r_char_len = len(r_str)
                             box_fill = ColorStr(self._AAbbrv[self.BestAction[tuple(idx)]].center(self._SquareSize), bg=('r' if rr<0 else 'g'))
                             self._RoadsMap[tuple(idx)] = box_fill
                        else:
                             self._RoadsMap[tuple(idx)] = self._AAbbrv[self.BestAction[tuple(idx)]].center(self._SquareSize)
                             self._BWRoadsMap[tuple(idx)]  = self._AAbbrv[self.BestAction[tuple(idx)]].center(self._SquareSize*0)
                   del idx;

         # clean_output = ('\n'.join([''.join(i) for i in self._BWRoadsMap])) # FIXME DEBUG
         formatted_output = ('\n'.join([''.join(i) for i in self._RoadsMap])) # This print out fine in python. messy if exported to text or figure
         if do_print: print(formatted_output)
         return self._BWRoadsMap, formatted_output

#####################################################################
    def  SolutionTrail(self, start_at = None, draw_on = None, do_print=True):

         if isinstance(draw_on,None.__class__):
              self._SolutionTrail = copy.deepcopy(self._GridMap)

         else:
              assert draw_on.shape == self._GridMap.shape, 'Provided grid/map to draw on has shape (%i, %i). Shape of the actual grid is (%i,%i)!' %(*draw_on.shape,*self._GridMap)
              self._SolutionTrail = copy.deepcopy(draw_on)

         self._BWSolutionTrail = copy.deepcopy(self._Grid)
         which_nan = (self._BWSolutionTrail!=self._BWSolutionTrail) # nan != nan
         self._BWSolutionTrail[which_nan] = False
         self._BWSolutionTrail[MyGrid.Walls] = u'\u2588'

         self.UpdateBestAction()
         itr_max = np.prod(self.Paths.shape)*3 # stop if you move 3 times more than total number of boxes

         if   isinstance(start_at,None.__class__):
              start_at = self._Start_Idx
         elif isinstance(start_at,(list,np.ndarray,tuple)):
              for start_point in self._Start_Idx: # Clear the "starting"
                   self._SolutionTrail[tuple(start_point)] = ''.center(self._SquareSize)
              if len(start_at) == 2:
                   start_at = np.array([start_at],dtype=tuple)
              else:
                   raise ValueError('start_at has ambiguous size %i' %len(start_at))

         for start_point in start_at: # if there are several starting points. though shouldn't happen
              # itr=0; start_point = self._Start_Idx[itr]
              idx=tuple(start_point.tolist());
              print('Printing Solution trail starting @(%2i,%2i)' %idx)
              self._SolutionTrail[idx] = ColorStr(chr(9830).center(self._SquareSize),bg='y',fg='r')

              do_continue = True
              itr=0;
              state_visited=[]
              while do_continue and (itr<itr_max):
                   itr+=1;
                   action = self._Actions_Idx2Nam[self.BestAction[idx].item()]
                   aabrv  = self._AAbbrv[self.BestAction[idx].item()]
                   action_out, newIdx, newR, nextV, isValid = self.Move(action, pos=idx, serious=False) # action_out should be = action
                   still  = (tuple(newIdx) == tuple(idx))
                   state_visited.append(idx);
                   if itr>1: # For first box (starting point), i already put the symbol chr(9830)
                        # self._SolutionTrail[idx] = ColorStr(aabrv.center(self._SquareSize), fg='y',bg='B')
                        self._SolutionTrail[idx] = ColorStr(aabrv.center(self._SquareSize), fg='G',bg='c')
                        self._BWSolutionTrail[idx] = aabrv

                   if self.Walls[idx] or self._Trmnl[idx] or (still==True) or (idx in state_visited[:-1]):
                        do_continue=False
                        action_is_sit_still = (self._Actions[self._Actions_Idx2Nam[self.BestAction[idx].item()]]==[0,0])
                        status_is_jackpot = self._Grid[idx] == self._Jackpot
                        self._SolutionTrail[idx] = ColorStr(aabrv.center(self._SquareSize), fg='y',bg='B')
                        if self._Trmnl[idx] or status_is_jackpot:
                             print('Solution found after %i transitions @(%2i,%2i)' %(itr,*idx))
                        else:
                             print('Stopped printing solution trail at box #(%2i,%2i)' %idx)
                        break
                   else:
                        idx = tuple(newIdx);

              if itr>=itr_max:
                   beep(F=9000,t=900)
                   print('Qutting printing solution trail after sweeping %i boxes. Something must be wrong' %itr)

         formatted_output = ('\n'.join([''.join(i) for i in self._SolutionTrail])) # This print out fine in python. messy if exported to text or figure
         if do_print: print(formatted_output)
         return formatted_output, self._SolutionTrail, self._BWSolutionTrail

#####################################################################

    def __repr__(self): # self;
        # [print(*i,sep='') for i in self._GridMap];
        return '\n'.join([''.join(i) for i in self._GridMap])

    def __str__(self): # print(self)
        # [print(*i,sep='') for i in self._GridMap];
        return self.__repr__()

#####################################################################
    @staticmethod
    def   EvaluateState(GridObject,*,idx,Transition_prob,Vs0,discount_factor=0.999):
          idx=tuple(idx)
          if GridObject._Trmnl[idx]: # This is Bug#2. I was treating terminal states like all other states
              # GridObject.V[(idx)] = GridObject._R[(idx)];
              # GridObject.V[(idx)] = GridObject._R[(idx)] * discount_factor / (1-discount_factor);
              V_new = GridObject._R[idx] / (1-discount_factor);
              Q_new  = np.zeros_like(GridObject._Actions_set,dtype=float)
              Q_new[GridObject._StillAction[0]] = V_new
          else:
              Expeditions    = np.asarray([GridObject.Move(x, pos=idx, serious=False) for x in GridObject._Actions_set],dtype=object)
              # if idx==(3,9): print(GridObject); print(GridObject.V_df); print('Stopped to debug @(%2i,%2i)' %idx); breakpoint();
              actions_out    = Expeditions[:,0];
              s_next         = Expeditions[:,1]
              Rewards        = Expeditions[:,2]
              # Vs0_next       = Expeditions[:,3] # I dont use this
              is_valid       = Expeditions[:,4]
              # Vs0_next        = np.vectorize(lambda x: 0 if x==None else Vs0[tuple(x)])(Expeditions[:,0]); # we V(s'), s' is the next state, not the current. i made a mistake here the first time i coded this
              Vs0_next        = [Vs0[tuple(x)] for x in Expeditions[:,1]]; # You have to use Vs0, not latest updated GridObject.V. This Bug#1
              UpdateSet       = np.asarray([r + discount_factor * v for r,v in zip(Rewards, Vs0_next)]); #i made a mistake here the first time i did this. i used: discount_factor * reward + V_old(s); V_old(s) is wrong. we need gamma * V_old(s')

              if False:
                 UpdateDict     = {i: np.average(UpdateSet[j],weights = Transition_prob[j]).item() \
                                   for i,j in GridObject._Actions_Nam2Idx.items()} # expectation  for each action. this NOT up to you. you can NOT take max here, because you can't control transition probability;
                 V_new  = np.average(list(UpdateDict.values()),
                                     weights = tuple( GridObject._Pi[idx] )).item()

              else:
                   UpdateDict     = {i:np.nansum(UpdateSet[j]*Transition_prob[j]).item() for i,j in GridObject._Actions_Nam2Idx.items()} # expectation  for each action. this NOT up to you. you can NOT take max here, because you can't control transition probability;
                   V_new  = np.nansum( np.array(list(UpdateDict.values())) * GridObject._Pi[idx] )

              Rewards *= 0 #just to reset
              Q_new  = list( UpdateDict.values() )
              # print('%sItr#%2i: V(%2i,%2i) = %6.4g, Reward = %s' %('\n' if idx==(1,1) else '', itr[1]+1,*idx,MyGrid.V[tuple(idx)],[(key,reward) for key,reward in zip(MyGrid._Actions.keys(),Rewards.tolist())]))

          return dict(idx=idx, V_new=V_new, Q_new=Q_new)

#####################################################################
    def  PolicyEvaluation(self,Transition_prob,Vs0,discount_factor=0.999):
          # Go through all states
          ############# Iterating over all states (the boxes in the grid) #############
          # assert 0<discount_factor<1, 'Discount factor == %g' %discount_factor
          if True:
               par_fun = lambda sym_idx: self.EvaluateState(self,
                                                            idx=sym_idx,
                                                            Transition_prob=Transition_prob,
                                                            Vs0=Vs0,
                                                            discount_factor=discount_factor)
               # par_fun = lambda sym_idx: sum(sym_idx); # Even this doesn't work. so thep problem is NOT with your function

               with Pool(n_cores//2) as parpool:
                    Updates = parpool.map(par_fun,self.Paths_idx) # par_fun must be defined beforehand. you can't define lambda inside MAP

               for update in Updates:
                    self.V[update.get('idx')] = update.get('V_new')
                    self.Q[update.get('idx')] = update.get('Q_new')

          else:
               for  itr_idx,idx in enumerate(self.Paths_idx): # itr_idx=0; idx=GridObject.Paths_idx[itr_idx]
                    update = self.EvaluateState(GridObject=self,
                                                idx=idx,
                                                Transition_prob=Transition_prob,
                                                Vs0=Vs0,
                                                discount_factor=discount_factor)
                    self.V[tuple(idx)] = update['V_new']
                    self.Q[tuple(idx)] = update['Q_new']

          del idx
          return  self.V, self.Q

#####################################################################
    @staticmethod
    def   ImproveState(GridObject,idx,epsilon):
          Expeditions     = np.asarray([GridObject.Move(x, pos=idx, serious=False) for x in GridObject._Actions_set],dtype=object)
          actions_out     = Expeditions[:,0];
          s_next          = Expeditions[:,1]
          Rewards         = Expeditions[:,2]
          Vs0_next        = Expeditions[:,3]
          is_valid        = Expeditions[:,4]
          GridObject.Q[tuple(idx)]  = Expeditions[:,3]
          GridObject.UpdateBestAction(idx=tuple(idx));
          # print('Best action @(%2i,%2i) is [%i] = [%s=%s] with value Q=%g ' %(*idx,
          #                                                                  bst_act:=GridObject.BestAction[tuple(idx)],
          #                                                                  GridObject._Actions_set[bst_act],
          #                                                                  GridObject._Actions_set[bst_act],
          #                                                                  GridObject.Q[tuple([*idx,bst_act])]))

          Pi_new  = np.ones_like(GridObject.Q[tuple(idx)]) * (epsilon/len(GridObject._Actions.keys())) # Pi(not-best) = epsilon
          Pi_new[GridObject.BestAction[tuple(idx)]] += (1-epsilon) # Pi(optimal) = 1-epsilon

          return dict(idx=idx, Pi_new=Pi_new)

#####################################################################
    def  PolicyImprovement(self,epsilon):
          for  itr_idx,idx in enumerate(self.Paths_idx): # itr_idx=0; idx=GridObject.Paths_idx[itr_idx]
               Update = self.ImproveState(self,idx,epsilon)
               self.Pi[tuple(idx)] = Update['Pi_new'];
               del Update;
               ########### Implement the policy ###########

          return self.Pi

#%% Import Grid as excel
Actions = {
            'L'     : [ 0,-1],
            'U'     : [-1, 0],
            'R'     : [ 0, 1],
            'D'     : [ 1, 0],
            'S'     : [ 0, 0], # Stay still
            #
            # 'UL'    : [-1,-1], #diagonal moves: Up left
            # 'DL'    : [ 1,-1],
            # 'UR'    : [-1, 1],
            # 'DR'    : [ 1, 1],
            #
            # '2L'    : [ 0,-2], #leap
            # '2D'    : [ 2, 0],
            # '2R'    : [ 0, 2],
            # '2U'    : [-2, 0],
            #
            # '2UL'   : [-2,-2], #leap diagonally
            # '2UR'   : [-2, 2],
            # '2DL'   : [ 2,-2],
            # '2DR'   : [ 2, 2],
            #
            # 'K1'    : [ 1,-2], #knight move
            # 'K2'    : [ 2,-1],
            # 'K3'    : [ 2, 1],
            # 'K4'    : [ 1, 2],
            # 'K5'    : [-1, 2],
            # 'K6'    : [-2, 1],
            # 'K7'    : [-2,-1],
            # 'K8'    : [-1,-2],
           }

ActionsPrint = {
            'S'     : 'o', # ord(u'\u2190') <===> chr(8592);    ord(u'\u2190')=8592
            'L'     : u'\u2190', # ord(u'\u2190') <===> chr(8592);    ord(u'\u2190')=8592
            'U'     : u'\u2191',
            'R'     : u'\u2192',
            'D'     : u'\u2193',
            #
            'UL'    : u'\u2b66', #diagonal moves: Up left
            'UR'    : u'\u2b67',
            'DR'    : u'\u2b68',
            'DL'    : u'\u2b69',
            #
            '2L'    : u'\u21d0', #leap
            '2U'    : u'\u21d1',
            '2R'    : u'\u21d2',
            '2D'    : u'\u21d3',
            #
            '2UL'   : u'\u21d6', #leap diagonally
            '2UR'   : u'\u21d7',
            '2DL'   : u'\u21d9',
            '2DR'   : u'\u21d8',
            #
            'K1'    : u'\u2ba0', #knight moves. Down 1 left 2
            'K4'    : u'\u2ba1', # down 1 right 2
            'K8'    : u'\u2ba2', # up 1 left 2
            'K5'    : u'\u2ba3', # up 1 right 2
            'K7'    : u'\u2ba4', # up 2 left 1
            'K6'    : u'\u2ba5', # up 2 right 1
            'K2'    : u'\u2ba6', # down 2 left 1
            'K3'    : u'\u2ba7', # down 2 right 1
           }

# Act_Probs = dict.fromkeys(Actions.keys(), 1/len(Actions));
#------------------------------------------------------------------------

InputData0=  \
[{
  'FileName'     : r"E:\LIST\Produce\Python practice\RL\GridWorld.xlsx",
  'Range'        : 'A:D',
    # 'Sheet'        : 'Sheet1',  # size 6   x 5   = 30
    # 'Sheet'        : 'Sheet2',  # size 4   x 4   = 16       basic example with completely random policy from sutton&barto
    # 'Sheet'        : 'Sheet3',  # size 7   x 7   = 49
    # 'Sheet'        : 'Sheet4',  # size 7   x 7   = 49       Shiyu Zhabo little maze
    # 'Sheet'        : 'Sheet5',  # size 6   x 10  = 60       snake maze
    # 'Sheet'        : 'Sheet6',  # size 6   x 10  = 60       cliff side walk
    # 'Sheet'        : 'Sheet7',  # size 12  x 19  = 228      Designed by me
    'Sheet'        : 'Sheet8',  # size 29  x 15  = 435      medium labyrinth with 2 solutions, one solution is shorter
    # 'Sheet'        : 'Sheet9',  # size 20  x 41  = 820
    # 'Sheet'        : 'Sheet10',  # size 31  x 31  = 961      Indian Book - Abhishek Nandy - RL with OpenAI, Keras and Tensorflow
    # 'Sheet'        : 'Sheet11', # size 121 x 61  = 7381     failed to solve
    # 'Sheet'        : 'Sheet12', # size 101 x 101 = 10201  solved with discount_factor=0.99
    # 'Sheet'        : 'Sheet13', # size 101 x 101 = 10201    solved with discount_factor=0.99
},
]

InputData = InputData0[0]
excelName, Range, Sheet = InputData.values();
print('Ignore the warning about \"conditional formatting\" below')
FullSheet               = pd.read_excel(excelName, header = None, sheet_name = Sheet) # Conditional formatting warning here
FullSheet               = FullSheet.astype(object).values

#------------------------------------------------------------------------

invalidPenalty=-1; # penalty for invalid move such as bumping into a wall. this was -1
V_init_case=0

if V_init_case==0:
     V_init = 0;
     invalidPenalty = 0 # hurts the terminal state

elif V_init_case==1: # Multiplly all rewards by 10
     V_init = copy.deepcopy(FullSheet)
     V_init_num = np.vectorize(lambda x: isinstance(x,(int,float,bool)))(V_init)*9 + 1
     V_init *= V_init_num

elif V_init_case==2: # reward**2
     V_init = copy.deepcopy(FullSheet)
     V_init = np.vectorize(lambda x: float(np.sign(x) * np.abs(x**2)) if isinstance(x,(int,float,bool)) else x,otypes=[object])(V_init)
     # V_init[V_init==-10]= -100 # For chinese book example Shiyu Zhao, at page 69 (82 / 283)

# pos_rewards = np.vectorize(lambda x: isinstance(x,(float,int,bool)) and (x>0))(FullSheet)
# neg_rewards = np.vectorize(lambda x: isinstance(x,(float,int,bool)) and (x<0))(FullSheet)

termination_condition = 0 # 0 works nice. 3 is wierd
# rewards             = np.vectorize(lambda x: x if (isinstance(x,(float,int,bool)) and (x<0 or x>0)) else 0)(FullSheet) # second condition eliminates NAN
rewards             = np.vectorize(lambda x: x if (isinstance(x,(float,int,bool)) and (x==x)) else 0)(FullSheet) # second condition eliminates NAN

if termination_condition==0: # Terminate at any reward (positive or negative)
     terminal       = rewards.astype(bool)
elif termination_condition==1: # Terminate at any negative reward
     terminal       = (rewards<0)
elif termination_condition==2: # Terminate at any positive reward
     terminal       = (rewards>0)
elif termination_condition==3: # Terminate at max positive reward only
     Trmnl_LinIdx   = np.flatnonzero(FullSheet == np.max(rewards,axis=None).item()).tolist()
     Trmnl_Idx      = np.vstack(np.unravel_index(Trmnl_LinIdx, rewards.shape)).T
     terminal       = np.zeros_like(FullSheet,dtype=bool)
     for  idx in Trmnl_Idx:
          terminal[tuple(idx.tolist())] = True

MyGrid = Grid(Map = FullSheet, Actions=Actions, ActionsPrint=ActionsPrint,
              V_init = V_init, invalidPenalty=invalidPenalty, terminal = terminal)
# MyGrid.Walls
# MyGrid.Paths
print('Imported maze from \"%s\". Maze has dimensions (%2i \u00D7 %2i)' %(Sheet,*MyGrid._Dims))
# manual initializaiton, to duplicate the example @9:55 in https://www.youtube.com/watch?v=Pka6Om0nYQ8

# sys.exit('Breakpoint after system definition')
#%% POLICY ITERATION ALGORITHM with epsilon-greedy policy
delta_target    = [5e-2, 1e-2] # [grand while loop - convergence of Pi     ,    inner loop - convergence of V(s) was 3e-3]
itr_max         = [30, 1e3]; # [grand while (Pi(s) update)       ,    policy evaluation (V(s) update)]
# updating the state value is very important for policy update. let the "policy evaluation" itr_max be large,
# if convergence is detected, the loop will stop BEFORE reaching itr_max
discount_factor = 0.95; # for lage mazes, this should NOT be 1. should be less.
eps_range       = [0.0, 0.8]; # the range of "epsilon" for epsilon-greedy policy. 0 means be always greedy
# def epsilon_greed(itr, eps_range, itr_max=itr_max[0]):
#      if np.ptp(eps_range)>0:
#           return min(eps_range) + np.ptp(eps_range) * np.exp(-7*itr / itr_max)
#      else:
#           return np.max(eps_range);
if True:
     epsilon_greed   = lambda itr, eps_range=eps_range, itr_max=itr_max[0] : np.exp(-7*itr / itr_max)*(np.ptp(eps_range)) + min(eps_range) if np.ptp(eps_range)>0 else np.max(eps_range);
else:
     epsilon_greed   = lambda itr, eps_range=eps_range, itr_max=itr_max[0] : np.pow((-itr / itr_max)+1, 4)*(np.ptp(eps_range)) + min(eps_range) if np.ptp(eps_range)>0 else np.max(eps_range);
itr              = [0,float(nan)] # just a counter. intuitively it must start from zero. NOT a guess/initialization of any parameter

isStable = dict(Policy=False,Vs=False)
# newfig = plt.figure()
# breakpoint()
# benchmark_idx=(3,2);
updates=[[[],[]]]
start_time = [time.time(),0]
print('');

# Transition_prob = np.ones_like(MyGrid._Actions_set).astype(int)/len(MyGrid._Actions_set); # Wrong.
Transition_prob = np.ones_like(MyGrid._Actions_set).astype(int); # Wrong.
while (itr[0] < itr_max[0]) and (isStable['Policy']==False):
    delta   = [1e6, 1e6]
    itr[0] += 1
    itr[1]  = 0
    isStable['Vs']=False
    # print('Starting outer iter#%i' %itr[1])

    ###################### POLICY EVALUATION STAGE ######################
    # Iterate till convergence
    # Actions_set  = list(Actions.keys());
    # Transition_prob = np.ones_like(MyGrid._Actions_set).astype(int); ###
    start_time[1] = time.time()
    while (itr[1] < itr_max[1]) and (isStable['Vs']==False):
        # print('Iteration #%i' %itr[0])
        assert not isStable['Vs'], 'somehow isStable[\'Vs\']==True'
        Vs0 = copy.deepcopy(MyGrid.V); # keep Vs0 as reference to detect convergence. update MyGrid.V
        # if itr[1]>=(itr_max[1]*0):
        #      breakpoint()
        ############################
        MyGrid.PolicyEvaluation(Vs0 = Vs0,
                                Transition_prob=Transition_prob,
                                discount_factor=discount_factor)
        ############################

        delta[1] = np.max(abs(MyGrid.V - Vs0))
        if   (itr[1] in np.ceil(np.arange(0, 1.1, 0.1) * itr_max[1]) or delta[1] < delta_target[1]): # condition for printing
             print('Evaluation  Iter#%4i/%i, MAX value  update size = %g' %(itr[1], itr_max[1], delta[1]))
             # updates[-1][1].append(([itr[1]+1,delta[1]]))

        if (delta[1] < delta_target[1]):
             print('Stopping Evaluation at iteration #%i because of convergence of V' %itr[1])
             isStable['Vs']=True
             beep(F=1000,t=10) #t in ms
          # break
        itr[1] += 1;

        # print('V(benchmark = (%2i,%2i)) = %g' %(*idx,MyGrid.V[benchmark_idx])) if 'benchmark_idx' in locals() else None

    # closed the while loop (loop till convergence)
    ###################### end of POLICY EVALUATION STAGE ######################
    ######################### POLICY IMPROVEMENT STAGE #########################
    # Go through all states. (No need to repeat this stage inside a while loop. we're not seeking a convergence)

    Pi0 = copy.deepcopy(MyGrid.Pi);
    epsilon = epsilon_greed(itr[0]).item()
    ############################
    MyGrid.PolicyImprovement(epsilon)
    ############################

    delta[0] = np.sum(abs(MyGrid.Pi - Pi0)).item()
    exec_time = time.time() - start_time[1]
    print('Improvement Iter#%3i/%i (%4.2f%%) lasted %6.3gs, SUM of policy updates = %g.\n%s\n' %(itr[0],
                                                                                                 itr_max[0],
                                                                                                 itr[0]/itr_max[0]*100,
                                                                                                 exec_time,
                                                                                                 delta[0],
                                                                                                 '-'*70))
    # updates[-1][0] = [(itr[0]+1,delta[0])]; updates.append([[],[]])
    if (delta[0] < delta_target[0]):
         print('Stopping at iteration #%i because of convergence of Pi' %itr[0])
         isStable['Policy']=True
         beep(F=5000,t=100) #t in ms
         # perhaps try learning rate here.

    ###################### end of POLICY IMPROVEMENT STAGE ######################

MyGrid.UpdateBestAction();
finish_time = time.time()
print('Finished Policy Iteration Algorithm of \"%s\" @%s within %s' %(Sheet, FU.now(), FU.time2hms(finish_time - start_time[0])))
print('itr_max = %s,\tdiscount_factor=%g,\t delta_target=%s' %(itr_max,discount_factor, delta_target))
print('#'*60)
beep(F=6000,t=300) #t in ms

#%% Print Results
#### https://stackoverflow.com/questions/26692946/changing-colours-of-pixels-of-plt-imshow-image
MyPrint = lambda *args: print(*args, sep='\n',end='\n'*2)
MyPrint('Grid',MyGrid);
# MyPrint('Values',MyGrid.V);
# MyPrint('Initial Values',pd.DataFrame(MyGrid.V0).iloc[1:-1,1:-1].map('{:,.2f}'.format));
MyPrint('Values',MyGrid.V_df);
# MyPrint('Values',f"{np.array2string(MyGrid.V[1:-1,1:-1].astype(float), formatter={'float': lambda x: f'{x:8.2f}'})}");
RoadsMapClean, RoadsMapFormatted = MyGrid.RoadsMap()
SolTrail = MyGrid.SolutionTrail(do_print=False)
MyPrint('Solution Trail',SolTrail[0]); # You have to call the function RoadsMap for _RoadsMap to be available
MyPrint('Solution Trail',MyGrid.SolutionTrail(draw_on = MyGrid._RoadsMap, do_print=False)[0]); # You have to call the function RoadsMap for _RoadsMap to be available

plt.close('all')
fig = plt.figure('Solution of GridWorld (%s)' %Sheet, figsize=(16,10))
ax1 = plt.subplot(2,1,1)
rgba_v = MyGrid.image
normalized = plt.Normalize(MyGrid.V[MyGrid.Walls.__invert__()].min(axis=None), MyGrid.V[MyGrid.Walls.__invert__()].max(axis=None))(MyGrid.V.astype(float))
# fig = plt.figure('State-Values Heat Map')
rgba_v[MyGrid.Paths] = plt.cm.RdYlGn(normalized)[MyGrid.Paths] # blue is close to the goal. red is far from the goal.
htmp = ax1.imshow(rgba_v,interpolation='None',aspect='auto');
htmp.axes.set_title('State-Values Heat Map')
# cbar = plt.colorbar(htmp,ax=htmp.axes, fraction=0.02, pad=0.02, extend='both')

ax2 = plt.subplot(2,1,2)
rgba_sol = MyGrid.image
# rgba_sol[MyGrid.Clear_Paths,:3] = 1,1,1 # Set squares to white
PathIndicators = [ActionsPrint[key] for key in MyGrid._Actions_set]
# for RGB in range(3):
rgba_sol[MyGrid.Clear_Paths,0] = 1-np.isin(SolTrail[2],PathIndicators)[MyGrid.Clear_Paths]
rgba_sol[MyGrid.Clear_Paths,1] = 1-np.isin(SolTrail[2],PathIndicators)[MyGrid.Clear_Paths]
# fig = plt.figure('Solution Trail')
htmp = ax2.imshow(rgba_sol,interpolation='None',aspect='auto');
htmp.axes.set_title('Solution Trail')
plt.tight_layout()


still=''
while False:
     RandomStartingPoints_id = np.random.choice(MyGrid.Clear_Paths.sum(axis=None),size=4,replace=False)
     for rand_start_point in RandomStartingPoints_id: # itr=0; rand_start_point=RandomStartingPoints_id[itr]
          start_idx = MyGrid.Clear_Paths_idx[rand_start_point]
          print('-'*60)
          MyPrint('Solution Trail starting @(%2i,%2i)' %tuple(start_idx) , MyGrid.SolutionTrail(start_at=start_idx,
                                                                                              draw_on = MyGrid._RoadsMap,
                                                                                              do_print=False)[0]); # You have to call the function RoadsMap for _RoadsMap to be available
     try:
          if   (MyGrid.size >= 800) and False:
               fileName, fileExt = os.path.splitext(excelName)
               excelOutName = excelName.replace(fileExt, '-AlgOut%s' %fileExt)
               with pd.ExcelWriter(excelOutName,engine="openpyxl",
                                   mode='a' if os.path.isfile(excelOutName) else 'w',
                                   if_sheet_exists = 'replace' if os.path.isfile(excelOutName) else None,
                                   ) as writer: # xlswriter does not support append mode. every time you open the excel file, it will start a new blank file
                   Vpd = pd.DataFrame(MyGrid.V)
                   Vpd.to_excel(writer,
                                sheet_name='%s-V(s)' %Sheet,
                                startrow=0,
                                startcol=0,
                                index=False,
                                header=False,
                                )
                   RoadsMapPD = pd.DataFrame(RoadsMapClean)
                   RoadsMapPD.style.apply(lambda x: ['text-align: center' for x in x]).to_excel(writer,
                                          sheet_name='%s-Map' %Sheet,
                                          startrow=0,
                                          startcol=0,
                                          index=True,
                                          header=True,
                                          )

               os.startfile(excelOutName)
     except PermissionError as remErr:
          if remErr.args[1] == "Permission denied":
               beep(t=150,F=250)
               input("\n\n\nThe Excel file is%s open you idiot. Close it! and hit enter\t\t" %still);
               still=' still'
     else:
          break

# plt.close('all')
# newfig = plt.figure(0,figsize=[max([*map(len,RoadsMap.split('\n'))])//10,len(RoadsMap.split('\n'))//5])
# newfig.text(0,0,"{}".format(RoadsMap))
# newfig.text(0,0,"{}".format(MyGrid._GridMapRaw))
# OutFileName = excelName.replace(os.path.splitext(excelName)[-1],'-%s.txt' %Sheet)
# FU.WriteTXT(OutFileName, RoadsMapFormatted,overwrite=True)
# FU.WriteTXT(OutFileName, RoadsMapClean,overwrite=True)
print('Finished solving and printing Maze on %s' %Sheet)
sys.exit('Manual stop')

#%% Monte-Carlo Basic

# Actions_set   = np.(list(Actions.keys()), p=list(Act_Probs.values()), size=N_rand_acts)
# Transition_prob  = np.ones_like(Actions_set).astype(int); #i wouldn't know transition. transition would reveal itself in the sample
get_ipython().magic('%cls')
MyPrint = lambda *args: print(*args, sep='\n',end='\n'*2)
MyPrint('Grid',MyGrid);

ChainLength    = int(200);
N_chains_each  = int(100);


RewMat = []
discount_factor = discount_factor if 'discount_factor' in locals() else 0.9

ValidStartingPoints = np.argwhere(MyGrid.Paths);
N_chains_total = len(MyGrid._Actions.keys()) * MyGrid.Paths.sum(axis=None).item() * N_chains_each
ChainsHeads = np.tile(np.vstack(list(MyGrid._Actions.keys())),(MyGrid.Paths.sum(axis=None) * N_chains_each,1))
MCChains = np.hstack([ChainsHeads,
                      np.random.choice(list(Actions.keys()), size=[N_chains_total,ChainLength])])

# StartPoint = np.tile(ValidStartingPoints,(N_chains_each*len(MyGrid._Actions.keys()),1))
StartPoint = np.vstack([np.tile(xx,((N_chains_each*len(MyGrid._Actions.keys()),1))) for xx in ValidStartingPoints])

for iter_chain, start_point in enumerate(StartPoint):
    print('Exploring chain #%3i/%i: starting @(%2i,%2i)' %(iter_chain, N_chains_total, *start_point))
    current_state  = start_point;
    cum_reward = 0;
    rew = 0;
    chain = MCChains[iter_chain,:]
    RewMat.append([])
    for iter_step, step in enumerate(chain):
        # iter_step = 0; step = chain[iter_step]
        next_state, rew, nextVal, isValid = MyGrid.Move(step, pos = current_state, serious=False);
        # RewMat[iter_step,iter_chain,0] = rew
        # RewMat[iter_step,iter_chain,1] = (RewMat[iter_step-1,iter_chain,1]*discount_factor + rew)
        # when iter_step=0; iter_step-1 will be -1, and it will take the last element in the column which should be 0.
        # cum_reward = cum_reward*discount_factor + rew # FIXME. not sure if previous rewards (earlier in the chain) should be discounted, or the opposite
        cum_reward += (rew*(discount_factor**iter_step))# FIXME. not sure if previous rewards (earlier in the chain) should be discounted, or the opposite
        RewMat[-1].append([rew,cum_reward])

        if isValid == False:
            print('Breaking chain #%3i/%i @step#%3i/%i @box(%2i,%2i) because of INvalid move' %(iter_chain,
                                                                                                N_chains_total,
                                                                                                iter_step,
                                                                                                ChainLength,
                                                                                                *next_state))
            # RewMat[iter_chain,iter_step+1:,0]=nan;
            # RewMat[iter_chain,iter_step+1:,1]=nan;
            break;
        else:
            # print('\t@step#%3i/%i, took action [%s] and moved from (%2i,%2i) --> (%2i,%2i) for a reward: %g' %(iter_step,ChainLength,step,*current_state,*next_state,rew))
            current_state = next_state

    print('Finished chain #%3i/%i @step#%3i/%i @box(%2i,%2i) with accumulative return=%g\n\n%s' %(iter_chain,
                                                                                                  N_chains_total,
                                                                                                  iter_step,
                                                                                                  ChainLength,
                                                                                                  *next_state,
                                                                                                  RewMat[-1][-1][-1],
                                                                                                  '-'*60))
    # TrueLen[iter_chain] = iter_step
TrueLen = [len(xx) for xx in RewMat]


Return_summary =  [RewMat[iter_chain][xx-1][1] for iter_chain,xx in enumerate(TrueLen)]

# for iter_chain,xx in enumerate(TrueLen):
#      print('chain#%3i has Length=%3i' %(iter_chain, xx))
#      Return_summary =  RewMat[iter_chain][xx-1][1]


plt.close('all')
figs=[]
axs=[]

figs.append(plt.figure())
axs.append(plt.gca())
sct=plt.scatter(TrueLen,Return_summary,s=3)
axs[-1].hlines(y=np.max(Return_summary), xmin=0, xmax=ChainLength, color='k',linestyles='-',label='Highest Return')
axs[-1].hlines(y=MyGrid._Jackpot, xmin=0, xmax=ChainLength, color='b',linestyles=':',label='Jackpot')
axs[-1].hlines(y=0, xmin=0, xmax=ChainLength, color='k',linestyles=':')
axs[-1].hlines(y=np.min(Return_summary), xmin=0, xmax=ChainLength, color='r',linestyles=':',label='Lowest Return')
axs[-1].vlines(x=ChainLength, ymin=np.min(Return_summary), ymax=np.max(Return_summary), color='k',linestyles='-',label='Chain Length Cap')
axs[-1].vlines(x=np.max(TrueLen), ymin=np.min(Return_summary), ymax=np.max(Return_summary), color='g',linestyles=':',label='Longest chain')
axs[-1].vlines(x=0, ymin=np.min(Return_summary), ymax=np.max(Return_summary), color='k',linestyles='-')
axs[-1].legend(loc='upper center')
axs[-1].set_title('Scatter plot of Reward vs. Chain Length'.title())
axs[-1].set_xlim(-1,ChainLength+2)
axs[-1].set_xlabel('Chain Length')
axs[-1].set_ylabel('Final Return')

figs.append(plt.figure())
axs.append(plt.gca())
hist1=plt.hist(TrueLen,np.linspace(0.1,1,20)*ChainLength)
axs[-1].set_title('Histogram plot of Chain Length'.title())
axs[-1].set_xlabel('Chain Length')

figs.append(plt.figure())
axs.append(plt.gca())
hist1=plt.hist(Return_summary,bins=50)
axs[-1].vlines(x=MyGrid._Jackpot, ymin=0, ymax=np.max(hist1[0]), color='k',linestyles=':')
axs[-1].set_ylim(0,np.sort(hist1[0])[-2])
axs[-1].set_title('Histogram plot of Final Return'.title())
axs[-1].set_xlabel('Final Return')


print('#'*80)
print('Finished %i MC chains.' %N_chains_total)
print('Longest chain:  #%5i with length %3i ended with cumulative return = %g.' %(idx:=np.argmax(TrueLen).item(),TrueLen[idx],RewMat[idx][-1][1]))
print('Shortest chain: #%5i with length %3i ended with cumulative return = %g.' %(idx:=np.argmin(TrueLen).item(),TrueLen[idx],RewMat[idx][-1][1]))
print('Richest chain:  #%5i with length %3i ended with cumulative return = %g.' %(idx:=np.argmax(Return_summary).item(),TrueLen[idx],RewMat[idx][-1][1]))

for lngth in [3,5,10,20,50,100]:
     which_chains = [itr for itr,x in enumerate(TrueLen) if x>=lngth]
     print('-'*50)
     print('%5i/%i (%6.2f%%) chains are more than %3i steps long.' %(len(which_chains),N_chains_total,len(which_chains)/N_chains_total*100,lngth))
     if len(which_chains)>0:
          print('Average return of these chains is %g.' %np.mean([Return_summary[x] for x in which_chains]))
          print('Worst   return of these chains is %g.' %np.min([Return_summary[x] for x in which_chains]))
          print('Average length of these chains is %g.' %np.mean([TrueLen[x] for x in which_chains]))
print('#'*80)

for reward in np.linspace(-0.5,2,21)*MyGrid._Jackpot: # reward = 10
     which_chains = [itr for itr,x in enumerate(Return_summary) if (x>=reward if reward>0 else x<=reward)]
     print('-'*50)
     print('%5i/%i (%6.2f%%) chains had a return %s %g.' %(len(which_chains),N_chains_total,len(which_chains)/N_chains_total*100,'>=' if reward>0 else '<=', reward))
     if len(which_chains)>0:
          print('Average return of these chains is %g.' %np.mean([Return_summary[x] for x in which_chains]))
          print('Average length of these chains is %g.' %np.mean([TrueLen[x] for x in which_chains]))

sys.exit('Manual exit')

#%%
'''
ActionsPrint =
          { 'L'     : u'\u2190', # ord(u'\u2190') <===> chr(8592);    ord(u'\u2190')=8592
            'U'     : u'\u2191',
            'R'     : u'\u2192',
            'D'     : u'\u2193',

            # 'UL'    : u'\u2b66', #diagonal moves: Up left
            # 'UR'    : u'\u2b67',
            # 'DR'    : u'\u2b68',
            # 'DL'    : u'\u2b69',

            # '2L'    : u'\u21d0', #leap
            # '2U'    : u'\u21d1',
            # '2R'    : u'\u21d2',
            # '2D'    : u'\u21d3',

            # '2UL'   : u'\u21d6', #leap diagonally
            # '2UR'   : u'\u21d7',
            # '2DL'   : u'\u21d9',
            # '2DR'   : u'\u21d8',

            # 'K1'    : u'\u2ba0', #knight moves. Down 1 left 2
            # 'K4'    : u'\u2ba1', # down 1 right 2
            # 'K8'    : u'\u2ba2', # up 1 left 2
            # 'K5'    : u'\u2ba3', # up 1 right 2
            # 'K7'    : u'\u2ba4', # up 2 left 1
            # 'K6'    : u'\u2ba5', # up 2 right 1
            # 'K2'    : u'\u2ba6', # down 2 left 1
            # 'K3'    : u'\u2ba7', # down 2 right 1
           }

u'\u2190' = ←                   ,	chr(8592) = ←
u'\u2191' = ↑                   ,	chr(8593) = ↑
u'\u2192' = →                   ,	chr(8594) = →
u'\u2193' = ↓                   ,	chr(8595) = ↓
u'\u2194' = ↔                   ,	chr(8596) = ↔
u'\u2195' = ↕                   ,	chr(8597) = ↕
u'\u2196' = ↖                   ,	chr(8598) = ↖
u'\u2197' = ↗                   ,	chr(8599) = ↗
u'\u2198' = ↘                   ,	chr(8600) = ↘
u'\u2199' = ↙                   ,	chr(8601) = ↙
u'\u21b0' = ↰                   ,	chr(8624) = ↰
u'\u21b1' = ↱                   ,	chr(8625) = ↱
u'\u21b2' = ↲                   ,	chr(8626) = ↲
u'\u21b3' = ↳                   ,	chr(8627) = ↳
u'\u21b4' = ↴                   ,	chr(8628) = ↴
u'\u21b5' = ↵                   ,	chr(8629) = ↵
u'\u21d0' = ⇐                   ,	chr(8656) = ⇐
u'\u21d1' = ⇑                   ,	chr(8657) = ⇑
u'\u21d2' = ⇒                   ,	chr(8658) = ⇒
u'\u21d3' = ⇓                   ,	chr(8659) = ⇓
u'\u21d4' = ⇔                   ,	chr(8660) = ⇔
u'\u21d5' = ⇕                   ,	chr(8661) = ⇕
u'\u21d6' = ⇖                   ,	chr(8662) = ⇖
u'\u21d7' = ⇗                   ,	chr(8663) = ⇗
u'\u21d8' = ⇘                   ,	chr(8664) = ⇘
u'\u21d9' = ⇙                   ,	chr(8665) = ⇙
u'\u2b00' = ⬀                   ,	chr(11008) = ⬀
u'\u2b01' = ⬁                   ,	chr(11009) = ⬁
u'\u2b02' = ⬂                   ,	chr(11010) = ⬂
u'\u2b03' = ⬃                   ,	chr(11011) = ⬃
u'\u2b04' = ⬄                   ,	chr(11012) = ⬄
u'\u2b05' = ⬅                   ,	chr(11013) = ⬅
u'\u2b06' = ⬆                   ,	chr(11014) = ⬆
u'\u2b07' = ⬇                   ,	chr(11015) = ⬇
u'\u2b08' = ⬈                   ,	chr(11016) = ⬈
u'\u2b09' = ⬉                   ,	chr(11017) = ⬉
u'\u2b0a' = ⬊                   ,	chr(11018) = ⬊
u'\u2b0b' = ⬋                   ,	chr(11019) = ⬋

# dont use these it is wider than the standard character. it misses up the maze
# highlight their line to see their width
u'\u2b60' = ⭠                   ,	chr(11104) = ⭠
u'\u2b61' = ⭡                   ,	chr(11105) = ⭡
u'\u2b62' = ⭢                   ,	chr(11106) = ⭢
u'\u2b63' = ⭣                   ,	chr(11107) = ⭣

u'\u2b64' = ⭤                   ,	chr(11108) = ⭤
u'\u2b65' = ⭥                   ,	chr(11109) = ⭥
u'\u2b66' = ⭦                   ,	chr(11110) = ⭦
u'\u2b67' = ⭧                   ,	chr(11111) = ⭧
u'\u2b68' = ⭨                   ,	chr(11112) = ⭨
u'\u2b69' = ⭩                   ,	chr(11113) = ⭩
u'\u2b6a' = ⭪                   ,	chr(11114) = ⭪
u'\u2b6b' = ⭫                   ,	chr(11115) = ⭫
u'\u2b6c' = ⭬                   ,	chr(11116) = ⭬
u'\u2b98' = ⮘                   ,	chr(11160) = ⮘
u'\u2b99' = ⮙                   ,	chr(11161) = ⮙
u'\u2b9a' = ⮚                   ,	chr(11162) = ⮚
u'\u2b9b' = ⮛                   ,	chr(11163) = ⮛
u'\u2b9c' = ⮜                   ,	chr(11164) = ⮜
u'\u2b9d' = ⮝                   ,	chr(11165) = ⮝
u'\u2b9e' = ⮞                   ,	chr(11166) = ⮞
u'\u2b9f' = ⮟                   ,	chr(11167) = ⮟
u'\u2ba0' = ⮠                   ,	chr(11168) = ⮠
u'\u2ba1' = ⮡                   ,	chr(11169) = ⮡
u'\u2ba2' = ⮢                   ,	chr(11170) = ⮢
u'\u2ba3' = ⮣                   ,	chr(11171) = ⮣
u'\u2ba4' = ⮤                   ,	chr(11172) = ⮤
u'\u2ba5' = ⮥                   ,	chr(11173) = ⮥
u'\u2ba6' = ⮦                   ,	chr(11174) = ⮦
u'\u2ba7' = ⮧                   ,	chr(11175) = ⮧
u'\u2bc5' = ⯅                   ,	chr(11205) = ⯅
u'\u2bc6' = ⯆                   ,	chr(11206) = ⯆
u'\u2bc7' = ⯇                   ,	chr(11207) = ⯇
u'\u2bc8' = ⯈                   ,	chr(11208) = ⯈

'''
