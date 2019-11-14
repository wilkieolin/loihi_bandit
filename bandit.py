import os
import nxsdk.api.n2a as nx
import numpy as np
import matplotlib.pyplot as plt
import re
from nxsdk.graph.monitor.probes import *
from nxsdk.graph.processes.phase_enums import Phase

class arm:
    def __init__(self, net, neuronsPerArm, weight, c_prototypes, s_prototypes, lrArgs):
        self.neuronsPerArm = neuronsPerArm

        #setup the learning rule
        self.learningRule = net.createLearningRule(**lrArgs)
        #create the excitatory connectin prototype
        self.exSynPrototype = nx.ConnectionPrototype(learningRule=self.learningRule, **s_prototypes['exh_kwargs'])

        #create the excitatory/inhibitory compartment groups and comparator
        self.exhGroup = net.createCompartmentGroup(size=self.neuronsPerArm, prototype=c_prototypes['noisy'])
        self.inhGroup = net.createCompartmentGroup(size=self.neuronsPerArm, prototype=c_prototypes['noisy'])
        self.comparator = net.createCompartmentGroup(size=self.neuronsPerArm, prototype=c_prototypes['comparator'])
        #create the arm's reward buffer
        self.buffer = net.createCompartment(prototype=c_prototypes['buffer'])

        #connect the plastic excitatory synapses
        self.exhSynapses = self.exhGroup.connect(self.comparator,
                                                prototype=self.exSynPrototype,
                                                weight=np.repeat(weight, self.neuronsPerArm) * np.identity(self.neuronsPerArm),
                                                connectionMask=np.identity(neuronsPerArm))

        #connect the non-learning inhibitory synapses
        self.inhSynapses = self.inhGroup.connect(self.comparator,
                                                prototype=s_prototypes['inh'],
                                                weight=np.repeat(50, self.neuronsPerArm) * np.identity(self.neuronsPerArm),
                                                connectionMask=np.identity(neuronsPerArm))

        #connect the buffer to the learning rule
        self.bufferConnection = self.buffer.connect(self.learningRule.reinforcementChannel, prototype=self.exSynPrototype)

        #create the reward spike generator (dummy)
        self.spikeGen = net.createSpikeGenProcess(1)
        self.spikeGen.addSpikes([0], [[1]])
        self.spikeGen.connect(self.buffer, prototype=s_prototypes['passthrough'])

        #set up the probes
        self.rewardProbe = self.bufferConnection.probe(nx.ProbeParameter.REWARD_TRACE)
        #don't the spike probe it or this will interfere with the SNIP counting spikes
        customSpikeProbeCond = SpikeProbeCondition(tStart=10000000)
        self.spikeProbe = self.comparator.probe(nx.ProbeParameter.SPIKE, customSpikeProbeCond)
        self.weightProbe = self.exhSynapses.probe(nx.ProbeParameter.SYNAPSE_WEIGHT)

    #def change_weight(weight):
        #TODO#



class bandit:
    def __init__(self, numArms=5, neuronsPerArm=1, votingEpoch=128, epochs=10):
        self.numArms = numArms
        self.neuronsPerArm = neuronsPerArm
        self.totalNeurons = numArms * neuronsPerArm
        self.votingEpoch = votingEpoch
        self.epochs = epochs

        #set default values for weights and probabilities
        self.probabilities = 100 * np.ones(numArms, dtype='int')
        self.weights = 100 * np.ones(numArms, dtype='int')

        #initialize the network
        self.net = nx.NxNet()
        self.arms = []

        self.lrArgs = {'dw': 'u0*r1',
                        'r1Impulse': 2,
                        'r1TimeConstant': 1,
                        'tEpoch': 2}

        #setup the necessary NX prototypes
        self._create_prototypes()
        #use these to create the arms whose spiking output will choose a potential reward
        for i in range(numArms):
            self._create_arm(self.weights[i])
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

    def _create_arm(self, weight):
        self.arms.append(arm(self.net, self.neuronsPerArm, weight, self.c_prototypes, self.s_prototypes, self.lrArgs))

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
        rewardChannel = self.board.createChannel(b'rewardChannel', "int", (self.epochs+1))
        rewardChannel.connect(self.snip, None)
        self.inChannels.append(dataChannel)
        self.inChannels.append(rewardChannel)

    def _create_prototypes(self):
        #setup compartment prototypes
        self.c_prototypes = {}

        #randomly spiking compartment
        self.c_prototypes['noisy'] = \
        nx.CompartmentPrototype(biasMant=0,
                                biasExp=0,
                                vThMant=4,
                                compartmentVoltageDecay=0,
                                compartmentCurrentDecay=0,

                                enableNoise=1,
                                randomizeVoltage=1,
                                randomizeCurrent=0,
                                noiseMantAtCompartment=2,
                                noiseExpAtCompartment=7,
                                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                                 logicalCoreId=1)

        #noisy comparator compartment
        self.c_prototypes['comparator'] = \
            nx.CompartmentPrototype(biasMant=0,
                                biasExp=0,
                                vThMant=100,
                                compartmentVoltageDecay=0,
                                compartmentCurrentDecay=2048,

                                enableNoise=1,
                                randomizeVoltage=0,
                                randomizeCurrent=1,
                                noiseMantAtCompartment=0,
                                noiseExpAtCompartment=7,
                                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                                 logicalCoreId=0)

        #compartment which can pass spikes for the reinforcement signal
        self.c_prototypes['buffer'] = \
            nx.CompartmentPrototype(biasMant=0,
                                biasExp=0,
                                vThMant=254,
                                compartmentVoltageDecay=0,
                                compartmentCurrentDecay=4095,

                                enableNoise=0,
                                randomizeVoltage=0,
                                randomizeCurrent=1,
                                noiseExpAtCompartment=7,
                                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                                 logicalCoreId=0)



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

    def get_probeid_map(self):
        pids = [[self.arms[j].spikeProbe[0].probes[i].n2Probe.counterId for i in range(self.neuronsPerArm)] for j in range(self.numArms)]

        return pids

    def run(self, epochs):
        assert epochs in range(1, self.epochs + 1), "Must run between 1 and the set number of epochs."
        dataChannel = self.inChannels[0]
        rewardChannel = self.inChannels[1]

        self.board.run(self.votingEpoch * epochs)
        choices = dataChannel.read(epochs)
        rewards = rewardChannel.read(epochs)

        return (choices, rewards)

    def stop(self):
        self.board.disconnect()

    def _send_config(self):
        rcLocations = self.get_reinforcement_channels()
        probeIDMap = self.get_probeid_map()

        #send the epoch length
        setupChannel = self.outChannels[0]
        setupChannel.write(1, [self.votingEpoch])

        #send reinforcementChannel locations
        for i in range(self.numArms):
            rcLoc = rcLocations[i]
            for j in range(4):
                setupChannel.write(1, [rcLoc[0][j]])

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
