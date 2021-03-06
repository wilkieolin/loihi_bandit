import os
import tracker
import prototypes
import nxsdk.api.n2a as nx
import numpy as np
import re
from nxsdk.graph.monitor.probes import *
from nxsdk.graph.processes.phase_enums import Phase

class conditional_bandit:
    def __init__(self, n_actions, n_states, **kwargs):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_estimates = n_actions * n_states
        self.n_per_state = kwargs.get("n_per_state", 1)
        self.l_epoch = kwargs.get("l_epoch", 128)
        self.n_epochs = kwargs.get("n_epochs", 100)
        self.epsilon = int(kwargs.get("epsilon", 0.1)*100)
        self.seed = kwargs.get("seed", 341257896)

        p_rewards = kwargs.get("p_rewards")
        if p_rewards is not None:
            assert p_rewards.shape == (self.n_states, self.n_actions), "Rewards must be in (n_states, n_actions) format."

            #convert 0-1 (probability) range to 0-100 (percentile, int)
            self.p_rewards = np.clip(p_rewards * 100, 0, 100).astype(np.int)
        else:
            self.p_rewards = np.random.randint(100, size=(self.n_states, self.n_actions))
        

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
        self._create_probes()

        #compile and link it to Loihi
        self._compile()
        self._create_SNIPs()
        self._create_channels()
    

    def _compile(self):
        self.compiler = nx.N2Compiler()
        self.board = self.compiler.compile(self.net)
        self.board.sync = True
    

    def _create_channels(self):
        assert hasattr(self, 'board'), "Must compile net to board before creating channels."
        assert hasattr(self, 'snip'), "Must add SNIP before creating channels."
        self.outChannels = {}
        self.inChannels = {}

        # need to send the epoch length, seed,
        # and for each state: probability (1) and counter compartment for each state
        # as well as the feedback (rwd/punishment/condition/state)
        n_outData = 3 + self.n_estimates * (1 + 3*4) + (self.n_states + self.n_actions + 2)*4

        setupChannel = self.board.createChannel(b'setupChannel', "int", n_outData)
        setupChannel.connect(None, self.snip)
        self.outChannels['setupChannel'] = setupChannel

        #create the data channels to return reward & choice at each epoch
        dataChannel = self.board.createChannel(b'dataChannel', "int", (self.n_epochs+1)*2)
        dataChannel.connect(self.snip, None)
        self.inChannels['dataChannel'] = dataChannel

        rewardChannel = self.board.createChannel(b'rewardChannel', "int", (self.n_epochs+1))
        rewardChannel.connect(self.snip, None)
        self.inChannels['rewardChannel'] = rewardChannel

        spikeChannel = self.board.createChannel(b'spikeChannel', "int", (self.n_epochs*self.n_estimates))
        spikeChannel.connect(self.snip, None)
        self.inChannels['spikeChannel'] = spikeChannel
    

    def _create_logic(self):
        self.compartments = {}
        self.connections = {}
        self.connection_maps = {}

        #create the buffer neurons which spike corresponding to state/condition/reward/punishment
        reward = self.net.createCompartmentGroup(size=1,
                                                prototype=self.c_prototypes['bufferProto'])
        punishment = self.net.createCompartmentGroup(size=1,
                                                prototype=self.c_prototypes['bufferProto'])
        state = self.net.createCompartmentGroup(size=self.n_actions,
                                                prototype=self.c_prototypes['bufferProto'])
        condition = self.net.createCompartmentGroup(size=self.n_states,
                                                prototype=self.c_prototypes['bufferProto'])

        self.compartments['reward'] = reward
        self.compartments['punishment'] = punishment
        self.compartments['state'] = state
        self.compartments['condition'] = condition

        #wire the stubs up to them
        rstub_to_reward = self.stubs['reward_stub'].connect(reward,
                                                            prototype=self.s_prototypes['spkconn'],
                                                            connectionMask=np.ones((1,1)))
        pstub_to_punish = self.stubs['punish_stub'].connect(punishment,
                                                            prototype=self.s_prototypes['spkconn'],
                                                            connectionMask=np.ones((1,1)))
        sstub_to_state = self.stubs['state_stubs'].connect(state,
                                                            prototype=self.s_prototypes['spkconn'],
                                                            connectionMask=np.eye(self.n_actions))
        cstub_to_cond = self.stubs['cond_stubs'].connect(condition,
                                                            prototype=self.s_prototypes['spkconn'],
                                                            connectionMask=np.eye(self.n_states))

        self.connections['rstub_to_reward'] = rstub_to_reward
        self.connections['pstub_to_punish'] = pstub_to_punish
        self.connections['sstub_to_state'] = sstub_to_state
        self.connections['cstub_to_cond'] = cstub_to_cond

        #create the condition-filtering ands
        rwd_ands = self.net.createCompartmentGroup(size=self.n_estimates,
                                                    prototype=self.c_prototypes['andProto'])
        pun_ands = self.net.createCompartmentGroup(size=self.n_estimates,
                                                    prototype=self.c_prototypes['andProto'])

        self.compartments['rwd_ands'] = rwd_ands
        self.compartments['pun_ands'] = pun_ands

        #wire them up to the reward/punishment/condition
        state_map = np.tile(np.eye(self.n_actions), self.n_states).transpose()
        condition_map = np.zeros((self.n_estimates, self.n_states))
        for i in range(self.n_states):
            inds = range(i*self.n_actions, (i+1)*self.n_actions)
            condition_map[inds,i] = 1
        

        self.connection_maps['condition_map'] = condition_map
        self.connection_maps['state_map'] = state_map

        condition_to_rwd = condition.connect(rwd_ands,
                                                prototype=self.s_prototypes['thirdconn'],
                                                connectionMask=condition_map)
        state_to_rwd = state.connect(rwd_ands,
                                        prototype=self.s_prototypes['thirdconn'],
                                        connectionMask=state_map)
        reward_to_rwd = reward.connect(rwd_ands,
                                        prototype=self.s_prototypes['thirdconn'],
                                        connectionMask=np.ones((self.n_estimates,1)))

        condition_to_pun = condition.connect(pun_ands,
                                                prototype=self.s_prototypes['thirdconn'],
                                                connectionMask=condition_map)
        state_to_pun = state.connect(pun_ands,
                                        prototype=self.s_prototypes['thirdconn'],
                                        connectionMask=state_map)
        punish_to_pun = punishment.connect(pun_ands,
                                            prototype=self.s_prototypes['thirdconn'],
                                            connectionMask=np.ones((self.n_estimates,1)))

        self.connections['condition_to_rwd'] =  condition_to_rwd
        self.connections['state_to_rwd'] =  state_to_rwd
        self.connections['reward_to_rwd'] = reward_to_rwd
        self.connections['condition_to_pun'] =  condition_to_pun
        self.connections['state_to_pun'] =  state_to_pun
        self.connections['punish_to_pun'] = punish_to_pun

        #wire the condition-filtering ands to the reward/punishment ands on the Q-trackers for each condition
        rwd_to_trackers = []
        pun_to_trackers = []
        and_mask = np.eye(self.n_estimates, self.n_estimates)
        self.connection_maps['and_mask'] = and_mask

        for i in range(self.n_states):
            inds = range(i*self.n_actions, (i+1)*self.n_actions)
            #the reward tracker we're connecting to
            tracker = self.trackers[i]
            #TODO - FLIP BACK TO REGULAR FOR NON-OPTIMISTIC TRACKER
            rwd_connection = rwd_ands.connect(tracker.stubs['istubs'],
                                            prototype=self.s_prototypes['spkconn'],
                                            connectionMask=and_mask[inds,:])
            rwd_to_trackers.append(rwd_connection)

            pun_connection = rwd_ands.connect(tracker.stubs['estubs'],
                                            prototype=self.s_prototypes['spkconn'],
                                            connectionMask=and_mask[inds,:])
            pun_to_trackers.append(pun_connection)
        
        self.connections['rwd_connection'] = rwd_connection
        self.connections['pun_connection'] = pun_connection

    

    def _create_probes(self):
        # -- Create Probes --
        self.probes = {}

        if self.recordSpikes:
            self.probes['rwd_ands'] = self.compartments['rwd_ands'].probe(nx.ProbeParameter.SPIKE)
            self.probes['pun_ands'] = self.compartments['pun_ands'].probe(nx.ProbeParameter.SPIKE)

            self.probes['rwd'] = self.compartments['reward'].probe(nx.ProbeParameter.SPIKE)
            self.probes['pun'] = self.compartments['punishment'].probe(nx.ProbeParameter.SPIKE)
            self.probes['state'] = self.compartments['state'].probe(nx.ProbeParameter.SPIKE)
            self.probes['condition'] = self.compartments['condition'].probe(nx.ProbeParameter.SPIKE)

    

    def _create_prototypes(self, vth):
        self.prototypes = prototypes.create_prototypes(self.vth)

        self.c_prototypes = self.prototypes['c_prototypes']
        self.n_prototypes = self.prototypes['n_prototypes']
        self.s_prototypes = self.prototypes['s_prototypes']
    

    def _create_SNIPs(self):
        assert hasattr(self, 'board'), "Must compile net to board before creating SNIP."
        includeDir = os.getcwd()
        self.snip = self.board.createSnip(Phase.EMBEDDED_MGMT,
                                     includeDir=includeDir,
                                     cFilePath = includeDir + "/cond_management.c",
                                     funcName = "run_cycle",
                                     guardName = "check")
    

    def _create_trackers(self):
        self.trackers = []
        #set up the q trackers for each condition
        for i in range(self.n_states):

            state_tracker = tracker.tracker(self.net,
                self.prototypes,
                self.n_actions,
                n_per_state=self.n_per_state,
                l_epoch=self.l_epoch,
                epsilon=self.epsilon,
                recordWeights=self.recordWeights,
                recordSpikes=self.recordSpikes)

            self.trackers.append(state_tracker)
    

    def _create_stubs(self):
        #create input stubs for SNIP to interface with the network
        self.stubs = {}

        reward_stub = self.net.createInputStubGroup(size=1)
        punish_stub = self.net.createInputStubGroup(size=1)
        state_stubs = self.net.createInputStubGroup(size=self.n_actions)
        cond_stubs = self.net.createInputStubGroup(size=self.n_states)

        self.stubs['reward_stub'] = reward_stub
        self.stubs['punish_stub'] = punish_stub
        self.stubs['state_stubs'] = state_stubs
        self.stubs['cond_stubs'] = cond_stubs
    

    def get_counter_locations(self):
        locs = []
        for i in range(self.n_states):
            for j in range(self.n_actions):
                compartmentId = self.trackers[i].compartments['counters'][j].nodeId
                compartmentLoc = self.net.resourceMap.compartmentMap[compartmentId]

                locs.append(compartmentLoc)

        return locs
    

    def get_stub_locations(self):
        locs = {}

        rewardAxonId = self.connections['rstub_to_reward'][0].inputAxon.nodeId
        locs['reward_axon'] = self.net.resourceMap.inputAxon(rewardAxonId)[0]

        punishAxonId = self.connections['pstub_to_punish'][0].inputAxon.nodeId
        locs['punish_axon'] = self.net.resourceMap.inputAxon(punishAxonId)[0]

        state_axons = []
        for i in range(self.n_actions):
            stateAxonId = self.connections['sstub_to_state'][i].inputAxon.nodeId
            state_axons.append(self.net.resourceMap.inputAxon(stateAxonId)[0])
        
        locs['state_axons'] = state_axons

        condition_axons = []
        for i in range(self.n_states):
            conditionAxonId = self.connections['cstub_to_cond'][i].inputAxon.nodeId
            condition_axons.append(self.net.resourceMap.inputAxon(conditionAxonId)[0])
        
        locs['condition_axons'] = condition_axons

        return locs
    

    def initialize(self):
        self._start()
        self._send_config()
    

    def run(self, epochs):
        assert epochs in range(1, self.n_epochs + 1), "Must run between 1 and the set number of epochs."

        #only reserve hardware once we actually need to run the network
        if not self.started:
            self.initialize()
            self.started = True

        dataChannel = self.inChannels['dataChannel']
        rewardChannel = self.inChannels['rewardChannel']
        spikeChannel = self.inChannels['spikeChannel']

        self.board.run(self.l_epoch * epochs)
        data = np.array(dataChannel.read(epochs*2)).reshape(epochs,2)
        self.choices = data[:,0]
        self.conditions = data[:,1]
        self.rewards = np.array(rewardChannel.read(epochs))
        singlewgt = self.s_prototypes['single'].weight
        self.spikes = np.array(spikeChannel.read(epochs*self.n_estimates), dtype='int').reshape(epochs, self.n_estimates)/(singlewgt*2**6)

        return (self.conditions, self.choices, self.rewards, self.spikes)
    

    def _send_config(self):
        #probeIDMap = self.get_probeid_map()
        stubLocations = self.get_stub_locations()
        counterLocations = self.get_counter_locations()

        #send the epoch length
        setupChannel = self.outChannels['setupChannel']
        setupChannel.write(1, [self.l_epoch])
        #write the epsilon value
        setupChannel.write(1, [self.epsilon])

        #send the random seed
        setupChannel.write(1, [self.seed])

        #send arm probabilities
        for i in range(self.n_estimates):
            setupChannel.write(1, [self.p_rewards.ravel()[i]])

        #send reward/punishment stub locations
        setupChannel.write(4, stubLocations['reward_axon'])
        setupChannel.write(4, stubLocations['punish_axon'])

        #send state stub locations
        for i in range(self.n_actions):
            stateAxon = stubLocations['state_axons'][i]
            setupChannel.write(4, stateAxon)

        #send condition stub locations
        for i in range(self.n_states):
            conditionAxon = stubLocations['condition_axons'][i]
            setupChannel.write(4, conditionAxon)

        #send the counter locations
        for i in range(self.n_estimates):
            setupChannel.write(4, counterLocations[i][:4])
    

    def _set_params_file(self):
        filename = os.getcwd()+'/cond_parameters.h'

        with open(filename) as f:
            data = f.readlines()

        f = open(filename, "w")
        for line in data:

            #update numarms
            m = re.match(r'^#define\s+N_ACTIONS', line)
            if m is not None:
                line = '#define N_ACTIONS ' + str(self.n_actions) + '\n'

            #update neuronsperarm
            m = re.match(r'^#define\s+N_STATES', line)
            if m is not None:
                line = '#define N_STATES ' + str(self.n_states) + '\n'

            f.write(line)

        f.close()
    

    def _start(self):
        assert hasattr(self, 'board') and hasattr(self, 'snip'), "Must have compiled board and snips before starting."
        self._set_params_file()
        self.board.startDriver()
    

    def stop(self):
        self.board.disconnect()
    

