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
        self.n_estimates = n_states * n_conditions
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
        #END

        self.recordWeights = kwargs.get('recordWeights', False)
        self.recordSpikes = kwargs.get('recordSpikes', False)

        self.net = nx.NxNet()
        self.vth = 255
        self.started = False

        #create the virtual network
        self._create_prototypes(self.vth)
        self._create_trackers()
        self._create_stubs()
        self._create_logic()

        #compile and link it to Loihi
        self._compile()
        self._create_SNIPs()
        self._create_channels()
    #END

    def _compile(self):
        self.compiler = nx.N2Compiler()
        self.board = self.compiler.compile(self.net)
        self.board.sync = True
    #END

    def _create_channels(self):
        assert hasattr(self, 'board'), "Must compile net to board before creating channels."
        assert hasattr(self, 'snip'), "Must add SNIP before creating channels."
        self.outChannels = []
        self.inChannels = []

        # need to send the epoch length, seed,
        # and for each state: probability (1) and counter compartment for each state
        # as well as the feedback (rwd/punishment/condition/state)
        n_outData = 3 + self.n_estimates * (1 + 3*4) + (self.n_conditions + self.n_states + 2)*4

        setupChannel = self.board.createChannel(b'setupChannel', "int", n_outData)
        setupChannel.connect(None, self.snip)
        self.outChannels.append(setupChannel)

        #create the data channels to return reward & choice at each epoch
        dataChannel = self.board.createChannel(b'dataChannel', "int", (self.epochs+1))
        dataChannel.connect(self.snip, None)
        self.inChannels.append(dataChannel)

        rewardChannel = self.board.createChannel(b'rewardChannel', "int", (self.epochs+1))
        rewardChannel.connect(self.snip, None)
        self.inChannels.append(rewardChannel)

        spikeChannel = self.board.createChannel(b'spikeChannel', "int", (self.epochs*self.numArms))
        spikeChannel.connect(self.snip, None)
        self.inChannels.append(spikeChannel)
    #END

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
        state_map = np.zeros((self.n_estimates, self.n_conditions))
        for i in range(self.n_conditions):
            state_map[i*n_states:(i+1)*n_states,i] = 1
        #END

        self.connection_maps['condition_map'] = condition_map
        self.connection_maps['state_map'] = state_map

        condition_to_rwd = self.stubs['cond_stubs'].connect(rwd_ands,
                                                    prototype=self.s_prototypes['thirdconn'],
                                                    connectionMask=condition_map)
        state_to_rwd = self.stubs['state_stubs'].connect(rwd_ands,
                                                    prototype=self.s_prototypes['thirdconn'],
                                                    connectionMask=state_map)
        reward_to_rwd = self.stubs['reward_stub'].connect(rwd_ands,
                                                    prototype=self.s_prototypes['thirdconn'],
                                                    connectionMask=np.ones((self.n_estimates,1)))

        condition_to_pun = self.stubs['cond_stubs'].connect(pun_ands,
                                                    prototype=self.s_prototypes['thirdconn'],
                                                    connectionMask=condition_map)
        state_to_pun = self.stubs['state_stubs'].connect(pun_ands,
                                                    prototype=self.s_prototypes['thirdconn'],
                                                    connectionMask=state_map)
        punish_to_pun = self.stubs['punish_stub'].connect(pun_ands,
                                                    prototype=self.s_prototypes['thirdconn'],
                                                    connectionMask=np.ones((self.n_estimates,1)))

        self.connections['condition_to_rwd'] =  condition_to_rwd
        self.connections['state_to_rwd'] =  state_to_rwd
        self.connections['condition_to_pun'] =  condition_to_pun
        self.connections['state_to_pun'] =  state_to_pun

        #wire the condition-filtering ands to the reward/punishment ands on the Q-trackers for each condition
        rwd_to_trackers = []
        pun_to_trackers = []
        and_mask = np.eye(self.n_conditions, self.n_conditions)
        self.connection_maps['and_mask'] = and_mask

        for i in range(self.n_conditions):
            range = i*self.n_states:(i+1)*self.n_states
            #the reward tracker we're connecting to
            tracker = self.trackers[i]
            rwd_connection = rwd_ands.connect(tracker.stubs['estubs']],
                                            prototype=self.s_prototypes['spkconn'],
                                            connectionMask=and_mask[range,:])
            rwd_to_trackers.append(rwd_connection)

            pun_connection = rwd_ands.connect(tracker.stubs['istubs']],
                                            prototype=self.s_prototypes['spkconn'],
                                            connectionMask=and_mask[range,:])
            pun_to_trackers.append(pun_connection)
        #END
        self.connections['rwd_connection'] = rwd_connection
        self.connections['pun_connection'] = pun_connection

    #END


    def _create_prototypes(self):
        self.prototypes = prototypes.create_prototypes(self.vth)

        self.c_prototypes = prototypes['c_prototypes']
        self.n_prototypes = prototypes['n_prototypes']
        self.s_prototypes = prototypes['s_prototypes']
    #END

    def _create_SNIPs(self):
        assert hasattr(self, 'board'), "Must compile net to board before creating SNIP."
        includeDir = os.getcwd()
        self.snip = self.board.createSnip(Phase.EMBEDDED_MGMT,
                                     includeDir=includeDir,
                                     cFilePath = includeDir + "/snips/cond_management.c",
                                     funcName = "run_cycle",
                                     guardName = "check")
    #END

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
    #END

    def _create_stubs(self):
        #create input stubs for SNIP to interface with the network
        self.stubs = {}

        reward_stub = self.net.createInputStub()
        punish_stub = self.net.createInputStub()
        state_stubs = self.net.createInputStubGroup(size=self.n_states)
        cond_stubs = self.net.createInputStubGroup(size=self.n_conditions)

        self.stubs['reward_stubs'] = reward_stub
        self.stubs['punish_stub'] = punish_stub
        self.stubs['state_stubs'] = state_stubs
        self.stubs['cond_stubs'] = cond_stubs
    #END
#END
