import os
import tracker
import prototypes
import nxsdk.api.n2a as nx
import numpy as np
import re
from nxsdk.graph.monitor.probes import *
from nxsdk.graph.processes.phase_enums import Phase

class conditional_bandit:
    def __init__(self, n_conditions, n_states, **kwargs):
        self.n_conditions = n_conditions
        self.n_states = n_states
        self.n_per_state = kwargs.get("n_per_state", 1)
        self.l_epoch = kwargs.get("l_epoch", 128)
        self.n_epochs = kwargs.get("n_epochs", 100)
        self.epsilon = kwargs.get("epsilon", 0.1)

        p_rewards = kwargs.get("p_rewards")
        if p_rewards is not None:
            assert p_rewards.shape == (self.n_conditions, self.n_states), "Rewards must be in (n_conditions, n_states) format."

            #convert 0-1 (probability) range to 0-100 (percentile, int)
            self.p_rewards = np.clip(p_rewards * 100, 0, 100).astype(np.int)
        else:
            self.p_rewards = np.random.randint(100, size=(self.n_conditions, self.n_states))

        self.recordWeights = kwargs.get('recordWeights', False)
        self.recordSpikes = kwargs.get('recordSpikes', False)

        self.net = nx.NxNet()
        self.vth = 255
        self.started = False

        self._create_prototypes(self.vth)
        self._create_trackers()
        self._create_stubs()
        self._create_logic()

    def _create_logic(self):
        #create the condition-filtering ands
        self.compartments = {}
        self.connections = {}
        self.connection_maps = {}

        andsPerCondition = self.n_states * self.n_conditions
        rwd_ands = self.net.createCompartmentGroup(size=andsPerCondition,
                                                    prototype=self.c_prototypes['andProto'])
        pun_ands = self.net.createCompartmentGroup(size=andsPerCondition,
                                                    prototype=self.c_prototypes['andProto'])

        self.compartments['rwd_ands'] = rwd_ands
        self.compartments['pun_ands'] = pun_ands

        #wire them up to the stubs
        condition_map = np.tile(np.eye(self.n_conditions), self.n_states).transpose()
        state_map = np.tile(np.eye(self.n_states), self.n_conditions).transpose()

        self.connection_maps['condition_map'] = condition_map
        self.connection_maps['state_map'] = state_map

        condition_to_rwd = self.stubs['cond_stubs'].connect(rwd_ands,
                                                    prototype=self.s_prototypes['halfconn'],
                                                    connectionMask=condition_map)
        state_to_rwd = self.stubs['state_stubs'].connect(rwd_ands,
                                                    prototype=self.s_prototypes['halfconn'],
                                                    connectionMask=state_map)

        condition_to_pun = self.stubs['cond_stubs'].connect(pun_ands,
                                                    prototype=self.s_prototypes['halfconn'],
                                                    connectionMask=condition_map)
        state_to_pun = self.stubs['state_stubs'].connect(pun_ands,
                                                    prototype=self.s_prototypes['halfconn'],
                                                    connectionMask=state_map)

        self.connections['condition_to_rwd'] =  condition_to_rwd
        self.connections['state_to_rwd'] =  state_to_rwd
        self.connections['condition_to_pun'] =  condition_to_pun
        self.connections['state_to_pun'] =  state_to_pun

    def _create_prototypes(self):
        self.prototypes = prototypes.create_prototypes(self.vth)

        self.c_prototypes = prototypes['c_prototypes']
        self.n_prototypes = prototypes['n_prototypes']
        self.s_prototypes = prototypes['s_prototypes']

    def _create_trackers(self):
        self.trackers = []
        #set up the q trackers for each condition
        for i in range(self.n_conditions):

            state_tracker = tracker.tracker(self.net,
                self.prototypes,
                n_states=self.n_states,
                n_per_state=self.n_per_state,
                l_epoch=self.l_epoch,
                epsilon=self.epsilon
                )

            trackers.append(state_tracker)

    def _create_stubs(self):
        #create input stubs for SNIP to interface with the network
        self.stubs = {}

        rwd_stubs = self.net.createInputStubGroup(size=self.n_states)
        pun_stubs = self.net.createInputStubGroup(size=self.n_states)
        cond_stubs = self.net.createInputStubGroup(size=self.n_conditions)

        self.stubs['rwd_stubs'] = rwd_stubs
        self.stubs['pun_stubs'] = pun_stubs
        self.stubs['cond_stubs'] = cond_stubs
