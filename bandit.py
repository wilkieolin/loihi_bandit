import os
import nxsdk.api.n2a as nx
import numpy as np
import matplotlib.pyplot as plt
import re
from nxsdk.graph.monitor.probes import *
from nxsdk.graph.processes.phase_enums import Phase

class arm:
    def __init__(self, net, **kwargs):
        self.neuronsPerArm = kwargs['neuronsPerArm']
        self.coreSliceInd = kwargs['coreSliceInd']
        self.net = net
        c_prototypes = kwargs['c_prototypes']
        s_prototypes = kwargs['s_prototypes']
        weight = kwargs['weight']
        lrArgs = kwargs['lrArgs']

        self._create_learning_rule(lrArgs)
        self._create_prototypes(c_prototypes, s_prototypes)
        self._create_compartments()
        self._create_connections(weight, s_prototypes)
        self._create_generators(s_prototypes)
        self._create_probes()

    def _create_compartments(self):
        #create the excitatory/inhibitory compartment groups and comparator
        self.exhGroup = self.net.createCompartmentGroup(size=self.neuronsPerArm, prototype=self.noisyCompPrototype)
        self.inhGroup = self.net.createCompartmentGroup(size=self.neuronsPerArm, prototype=self.noisyCompPrototype)
        self.comparator = self.net.createCompartmentGroup(size=self.neuronsPerArm, prototype=self.comparatorCompPrototype)
        #create the arm's reward buffer
        self.buffer =self.net.createCompartment(prototype=self.bufferCompPrototype)

    def _create_connections(self, weight, s_prototypes):
        #connect the plastic excitatory synapses
        self.exhSynapses = self.exhGroup.connect(self.comparator,
                                                prototype=self.exSynPrototype,
                                                weight=np.repeat(weight, self.neuronsPerArm) * np.identity(self.neuronsPerArm),
                                                connectionMask=np.identity(self.neuronsPerArm))

        #connect the non-learning inhibitory synapses
        self.inhSynapses = self.inhGroup.connect(self.comparator,
                                                prototype=s_prototypes['inh'],
                                                weight=np.repeat(50, self.neuronsPerArm) * np.identity(self.neuronsPerArm),
                                                connectionMask=np.identity(self.neuronsPerArm))
        #connect the buffer to the learning rule
        self.bufferConnection = self.buffer.connect(self.learningRule.reinforcementChannel, prototype=self.exSynPrototype)

    def _create_generators(self, s_prototypes):
        #create the reward spike generator (dummy)
        self.spikeGen = self.net.createSpikeGenProcess(1)
        #self.spikeGen.addSpikes([0], [[1]])
        self.spikeGen.connect(self.buffer, prototype=s_prototypes['passthrough'])

    def _create_learning_rule(self, lrArgs):
        #setup the learning rule
        self.learningRule = self.net.createLearningRule(**lrArgs)

    def _create_probes(self):
        #set up the probes
        self.rewardProbe = self.bufferConnection.probe(nx.ProbeParameter.REWARD_TRACE)
        #don't the spike probe it or this will interfere with the SNIP counting spikes
        customSpikeProbeCond = SpikeProbeCondition(tStart=10000000)
        self.spikeProbe = self.comparator.probe(nx.ProbeParameter.SPIKE, customSpikeProbeCond)
        self.weightProbe = self.exhSynapses.probe(nx.ProbeParameter.SYNAPSE_WEIGHT)


    def _create_prototypes(self, c_prototypes, s_prototypes):
        #create the compartment prototypes
        self.noisyCompPrototype = nx.CompartmentPrototype(**c_prototypes['noisy_kwargs'], logicalCoreId = self.coreSliceInd * 2)
        self.comparatorCompPrototype = nx.CompartmentPrototype(**c_prototypes['comparator_kwargs'], logicalCoreId = self.coreSliceInd * 2 + 1)
        self.bufferCompPrototype = nx.CompartmentPrototype(**c_prototypes['buffer_kwargs'], logicalCoreId = self.coreSliceInd * 2 + 1)
        #create the excitatory connection prototype
        self.exSynPrototype = nx.ConnectionPrototype(learningRule=self.learningRule, **s_prototypes['exh_kwargs'])

    #def change_weight(weight):
        #TODO#



class bandit:
    def __init__(self, numArms=5, neuronsPerArm=1, votingEpoch=128, epochs=10, **kwargs):
        self.numArms = numArms
        self.neuronsPerArm = neuronsPerArm
        self.totalNeurons = numArms * neuronsPerArm
        self.votingEpoch = votingEpoch
        self.epochs = epochs

        #set default values for weights and probabilities
        if 'probabilities' in kwargs:
            probs = kwargs['probabilities']
            for p in probs:
                assert p in range(0,100+1), "Probabilitiy must be in range [0,100]."
            assert len(probs) == self.numArms, "Must have probability for each arm."

            self.probabilities = np.array(probs, dtype='int')
        else:
            self.probabilities = 100 * np.ones(numArms, dtype='int')

        if 'weights' in kwargs:
            weights = kwargs['weights']
            assert len(weights), "Length of supplied weights must be equal to number of arms."

            self.weights = np.array(weights, dtype='int')
        else:
            self.weights = 100 * np.ones(numArms, dtype='int')

        #initialize the network
        self.net = nx.NxNet()
        self.arms = []

        self.lrArgs = {'dw': '-1*r1*u0',
                        'r1Impulse': 2,
                        'r1TimeConstant': 1,
                        'tEpoch': 2,
                        'printDebug': True}

        #setup the necessary NX prototypes
        self._create_prototypes()
        #use these to create the arms whose spiking output will choose a potential reward
        for i in range(numArms):
            self._create_arm(self.weights[i], i)
        #compile the generated network to a board
        self._compile()
        #create the SNIP which will select an arm, generate rewards, and communicate to
        #the reinforcement channels
        self._create_SNIPs()
        #create channels to/from the SNIP to read out the network's choices/rewards on host
        self._create_channels()
        #start the network on Loihi
        self._start()
        #send the configuration information required by the SNIP
        self._send_config()


    def _compile(self):
        self.compiler = nx.N2Compiler()
        self.board = self.compiler.compile(self.net)

    def _create_arm(self, weight, coreSliceInd):
        self.arms.append(arm(self.net,
            coreSliceInd = coreSliceInd,
            neuronsPerArm = self.neuronsPerArm,
            weight = weight,
            c_prototypes = self.c_prototypes,
            s_prototypes = self.s_prototypes,
            lrArgs = self.lrArgs))

    def _create_channels(self):
        assert hasattr(self, 'board'), "Must compile net to board before creating channels."
        assert hasattr(self, 'snip'), "Must add SNIP before creating channels."
        self.outChannels = []
        self.inChannels = []

        # need to send the epoch length, probability and reinforcement channel for each arm,
        # and the probeID <-> neuron map
        n_outData = 1 + self.numArms * 5 + self.totalNeurons
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


    def _create_prototypes(self):
        #setup compartment prototypes
        self.c_prototypes = {}

        #randomly spiking compartment
        self.c_prototypes['noisy_kwargs'] = \
            {'biasMant': 0,
            'biasExp': 0,
            'vThMant':4,
            'compartmentVoltageDecay': 0,
            'compartmentCurrentDecay': 0,

            'enableNoise': 1,
            'randomizeVoltage': 1,
            'randomizeCurrent': 0,
            'noiseMantAtCompartment': 2,
            'noiseExpAtCompartment': 7,
            'functionalState': nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE}

        #noisy comparator compartment
        self.c_prototypes['comparator_kwargs'] = \
            {'biasMant': 0,
            'biasExp': 0,
            'vThMant': 100,
            'compartmentVoltageDecay': 0,
            'compartmentCurrentDecay': 2048,

            'enableNoise': 1,
            'randomizeVoltage': 0,
            'randomizeCurrent': 1,
            'noiseMantAtCompartment': 0,
            'noiseExpAtCompartment': 7,
            'functionalState': nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE}

        #compartment which can pass spikes for the reinforcement signal
        self.c_prototypes['buffer_kwargs'] = \
            {'biasMant': 0,
            'biasExp': 0,
            'vThMant':254,
            'compartmentVoltageDecay': 0,
            'compartmentCurrentDecay': 4095,

            'enableNoise': 0,
            'randomizeVoltage': 0,
            'randomizeCurrent': 1,
            'noiseMantAtCompartment': 0,
            'noiseExpAtCompartment': 7,
            'functionalState': nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE}

        self.s_prototypes = {}

        self.s_prototypes['exh_kwargs'] = {'weight': 2,
                                    'delay': 0,
                                    'enableLearning': 1,
                                    'numTagBits': 8,
                                    'signMode': nx.SYNAPSE_SIGN_MODE.MIXED}

        self.s_prototypes['inh'] = nx.ConnectionPrototype(weight=2,
                                    delay=0,
                                    signMode=nx.SYNAPSE_SIGN_MODE.MIXED)

        self.s_prototypes['passthrough'] = nx.ConnectionPrototype(weight=255,
                                    delay=0,
                                    signMode=nx.SYNAPSE_SIGN_MODE.MIXED)


    def _create_SNIPs(self):
        assert hasattr(self, 'board'), "Must compile net to board before creating SNIP."
        includeDir = os.getcwd()
        self.snip = self.board.createSnip(Phase.EMBEDDED_MGMT,
                                     includeDir=includeDir,
                                     cFilePath = includeDir + "/management.c",
                                     funcName = "run_cycle",
                                     guardName = "check")

    def get_reinforcement_channels(self):
        assert hasattr(self, 'board'), "Must compile net to board before collecting reinforcement channels."
        rcLocations = [self.net.resourceMap.reinforcementChannel(i) for i in range(self.numArms)]

        return rcLocations

    def get_buffer_locations(self):
        return [self.net.resourceMap.compartment(self.arms[i].buffer.nodeId) for i in range(self.numArms)]

    def get_reward_probes(self):
        return [arm.rewardProbe for arm in self.arms]

    def get_probeid_map(self):
        pids = [[self.arms[j].spikeProbe[0].probes[i].n2Probe.counterId for i in range(self.neuronsPerArm)] for j in range(self.numArms)]

        return pids

    def get_weights(self):
        wps = [arm.weightProbe for arm in self.arms]
        n_d = len(wps[0][0][0].data)
        ws = np.zeros((self.numArms, self.neuronsPerArm, n_d), dtype='int')
        for i in range(self.numArms):
            for j in range(self.neuronsPerArm):
                ws[i,j,:] = wps[i][0][0].data

        return ws

    def run(self, epochs):
        assert epochs in range(1, self.epochs + 1), "Must run between 1 and the set number of epochs."
        dataChannel = self.inChannels[0]
        rewardChannel = self.inChannels[1]
        spikeChannel = self.inChannels[2]

        self.board.run(self.votingEpoch * epochs)
        choices = dataChannel.read(epochs)
        rewards = rewardChannel.read(epochs)
        spikes = np.array(spikeChannel.read(epochs*self.numArms), dtype='int').reshape(epochs, self.numArms)

        return (choices, rewards, spikes)

    def stop(self):
        self.board.disconnect()

    def _send_config(self):
        rcLocations = self.get_reinforcement_channels()
        probeIDMap = self.get_probeid_map()

        #TEST
        bufferLocations = self.get_buffer_locations()

        #send the epoch length
        setupChannel = self.outChannels[0]
        setupChannel.write(1, [self.votingEpoch])

        # #send reinforcementChannel locations
        # for i in range(self.numArms):
        #     rcLoc = rcLocations[i]
        #     for j in range(4):
        #         setupChannel.write(1, [rcLoc[0][j]])

        #TEST#
        #send buffer locations instead
        for i in range(self.numArms):
            bufferLoc = bufferLocations[i]
            for j in range(4):
                setupChannel.write(1, [bufferLoc[j]])

        #send arm probabilities
        for i in range(self.numArms):
            setupChannel.write(1, [self.probabilities[i]])

        #send probe map
        for i in range(self.numArms):
            for j in range(self.neuronsPerArm):
                setupChannel.write(1, [probeIDMap[i][j]])


    def _start(self):
        assert hasattr(self, 'board') and hasattr(self, 'snip'), "Must have compiled board and snips before starting."
        self.set_params_file(self.numArms, self.neuronsPerArm)
        self.board.startDriver()


    def set_params_file(self, numArms, neuronsPerArm):
        filename = os.getcwd()+'/parameters.h'

        with open(filename) as f:
            data = f.readlines()

        f = open(filename, "w")
        for line in data:

            #update numarms
            m = re.match(r'^#define\s+NUMARMS', line)
            if m is not None:
                line = '#define NUMARMS ' + str(numArms) + '\n'

            #update neuronsperarm
            m = re.match(r'^#define\s+NEURONSPERARM', line)
            if m is not None:
                line = '#define NEURONSPERARM ' + str(neuronsPerArm) + '\n'

            f.write(line)

        f.close()
