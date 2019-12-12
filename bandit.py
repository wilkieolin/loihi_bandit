import os
import nxsdk.api.n2a as nx
import numpy as np
import matplotlib.pyplot as plt
import re
from nxsdk.graph.monitor.probes import *
from nxsdk.graph.processes.phase_enums import Phase

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

        if 'seed' in kwargs:
            self.seed = kwargs['seed']
        else:
            self.seed = 329801

        if 'recordWeights' in kwargs:
            self.recordWeights = kwargs['recordWeights']
        else:
            self.recordWeights = False

        #initialize the network
        self.net = nx.NxNet()
        self.vth = 255
        self.started = False

        #setup the necessary NX prototypes
        self._create_prototypes()
        #use these to create the arms whose spiking output will choose a potential reward
        self._create_compartments()
        self._create_connections()
        self._create_probes()
        #compile the generated network to a board
        self._compile()
        #create the SNIP which will select an arm, generate rewards, and communicate to
        #the reinforcement channels
        self._create_SNIPs()
        #create channels to/from the SNIP to read out the network's choices/rewards on host
        self._create_channels()

    def _compile(self):
        self.compiler = nx.N2Compiler()
        self.board = self.compiler.compile(self.net)
        self.board.sync = True

    def _create_arm(self):
        self.arms.append(arm(self,
            self.net,
            neuronsPerArm = self.neuronsPerArm,
            recordWeights = self.recordWeights))

    def _create_channels(self):
        assert hasattr(self, 'board'), "Must compile net to board before creating channels."
        assert hasattr(self, 'snip'), "Must add SNIP before creating channels."
        self.outChannels = []
        self.inChannels = []

        # need to send the epoch length, seed, probability (numArms), and buffer id for each arm,
        # and the probeID <-> neuron map
        n_outData = 2 + self.numArms * (1 + 4) + self.totalNeurons
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

        self.noisekwargs = {'randomizeVoltage': 1,
            'randomizeCurrent': 0,
            'noiseMantAtCompartment': 0,
            'noiseExpAtCompartment': 7}

        #soma 'output' compartment
        self.c_prototypes['soma'] = nx.CompartmentPrototype(
            vThMant = self.vth,
            compartmentCurrentDecay = 4095,
            compartmentVoltageDecay = 0,
            thresholdBehavior = 0,
            functionalState = 2,
            enableNoise = 0,
            **self.noisekwargs)

        #voltage compartment, soft reset performed here
        self.c_prototypes['volt'] = nx.CompartmentPrototype(
            vThMant = self.vth,
            compartmentCurrentDecay = 4095,
            compartmentVoltageDecay = 0,
            thresholdBehavior = 2,
            functionalState = 2,
            enableNoise = 0,
            **self.noisekwargs)

        #memory compartment (current driver for volt)
        self.c_prototypes['memory'] = nx.CompartmentPrototype(
            vThMant = self.vth,
            compartmentCurrentDecay = 4095,
            compartmentVoltageDecay = 0,
            thresholdBehavior = 2,
            functionalState = 2,
            enableNoise = 0,
            **self.noisekwargs)

        #compartment to receive reward inputs from the SNIP
        self.c_prototypes['input'] = nx.CompartmentPrototype(
            biasMant = 1,
            biasExp = 6,
            vThMant = 50,
            compartmentCurrentDecay = 4095,
            compartmentVoltageDecay = 0,
            thresholdBehavior = 0,
            functionalState = 2,
            enableNoise = 0,
            **self.noisekwargs)

        self.s_prototypes = {}

        self.s_prototypes['exc'] = nx.ConnectionPrototype(weight = 2)

        self.s_prototypes['inh'] = nx.ConnectionPrototype(weight = -2)

        self.s_prototypes['vth_neg'] = nx.ConnectionPrototype(weight = -1*self.vth)

        self.s_prototypes['reward'] = nx.ConnectionPrototype(weight = 255)

        #create the multi-compartment neuron prototype
        self.c_prototypes['soma'].addDendrite([self.c_prototypes['volt']], nx.COMPARTMENT_JOIN_OPERATION.OR)
        self.c_prototypes['volt'].addDendrite([self.c_prototypes['memory']], nx.COMPARTMENT_JOIN_OPERATION.ADD)

        self.n_prototypes = {}
        self.n_prototypes['tracker'] = nx.NeuronPrototype(self.c_prototypes['soma'])

    def _create_compartments(self):
        self.stubs = self.net.createInputStubGroup(size=self.numArms)
        self.inputs = self.net.createCompartmentGroup(size=self.numArms, prototype=self.c_prototypes['input'])

        self.tracker = self.net.createNeuronGroup(size=self.totalNeurons, prototype=self.n_prototypes['tracker'])
        self.soma = self.tracker.soma
        self.integrator = self.tracker.dendrites[0]
        self.memory = self.tracker.dendrites[0].dendrites[0]

        #self.output = self.net.createCompartmentGroup(size=self.numArms)

    def _create_connections(self):
        self.connections = {}
        #make the self-reset connections
        self.connections['vth_reset'] = self.soma.connect(self.integrator,
            prototype = self.s_prototypes['vth_neg'],
            connectionMask = np.identity(self.numArms))
        #make the reward connections
        cm = np.zeros((self.numArms, self.totalNeurons), dtype='int')
        for i in range(self.numArms):
            cm[i, (i*self.neuronsPerArm):((i+1)*self.neuronsPerArm)] = 1

        self.connections['reward'] = self.inputs.connect(self.memory,
            prototype=self.s_prototypes['exc'],
            connectionMask=cm)

        #make the buffer connections which can drive rewards & receive signals from SNIP
        self.connections['stubs'] = self.stubs.connect(self.inputs, prototype=self.s_prototypes['reward'], connectionMask=np.identity(self.numArms))

    def _create_probes(self):
        customSpikeProbeCond = SpikeProbeCondition(tStart=100000000)
        self.spikeProbe = self.soma.probe(nx.ProbeParameter.SPIKE, customSpikeProbeCond)
        #self.vSpkProbe = self.integrator.probe(nx.ProbeParameter.SPIKE)
        #self.rwdProbe = self.inputs.probe(nx.ProbeParameter.SPIKE)
        if self.recordWeights:
            self.weightProbe = self.memory.probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE)
            self.vProbe = self.integrator.probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE)

    def _create_SNIPs(self):
        assert hasattr(self, 'board'), "Must compile net to board before creating SNIP."
        includeDir = os.getcwd()
        self.snip = self.board.createSnip(Phase.EMBEDDED_MGMT,
                                     includeDir=includeDir,
                                     cFilePath = includeDir + "/management.c",
                                     funcName = "run_cycle",
                                     guardName = "check")

    def get_buffer_locations(self):
        locs = []
        for i in range(self.numArms):
            axonId = self.connections['stubs'][i].inputAxon.nodeId
            locs.append(self.net.resourceMap.inputAxon(axonId)[0])
        return locs

    def get_buffer_spikes(self):
        return np.array(self.rwdProbe[0].data)

    def get_mean_optimal_action(self):
        assert hasattr(self, 'choices'), "Must run network to get MOA."
        bestarm = np.argmax(self.probabilities)
        #TODO finish

    def get_probeid_map(self):
        pids = [self.spikeProbe[0][i].n2Probe.counterId for i in range(self.totalNeurons)]

        return pids

    def get_weights(self):
        assert self.recordWeights, "Weights were not recorded."

        n_d = len(self.weightProbe[0][0].data)
        ws = np.zeros((self.totalNeurons, n_d), dtype='int')
        for i in range(self.totalNeurons):
            ws[i,:] = self.weightProbe[0][i].data

        return ws

    def get_voltages(self):
        assert self.recordWeights, "Weights were not recorded."

        n_d = len(self.vProbe[0][0].data)
        vs = np.zeros((self.totalNeurons, n_d), dtype='int')
        for i in range(self.totalNeurons):
            vs[i,:] = self.vProbe[0][i].data

        return vs

    def init(self):
        self._start()
        self._send_config()

    def run(self, epochs):
        #only reserve hardware once we actually need to run the network
        if not self.started:
            self.init()
            self.started = True

        assert epochs in range(1, self.epochs + 1), "Must run between 1 and the set number of epochs."
        dataChannel = self.inChannels[0]
        rewardChannel = self.inChannels[1]
        spikeChannel = self.inChannels[2]

        self.board.run(self.votingEpoch * epochs)
        self.choices = np.array(dataChannel.read(epochs))
        self.rewards = np.array(rewardChannel.read(epochs))
        self.spikes = np.array(spikeChannel.read(epochs*self.numArms), dtype='int').reshape(epochs, self.numArms)

        return (self.choices, self.rewards, self.spikes)

    def reset(self):
        self.board.fetch()


    def stop(self):
        self.board.disconnect()

    def _send_config(self):
        probeIDMap = self.get_probeid_map()
        bufferLocations = self.get_buffer_locations()

        #send the epoch length
        setupChannel = self.outChannels[0]
        setupChannel.write(1, [self.votingEpoch])

        #send the random seed
        setupChannel.write(1, [self.seed])

        # #send reinforcementChannel locations
        # for i in range(self.numArms):
        #     rcLoc = rcLocations[i]
        #     for j in range(4):
        #         setupChannel.write(1, [rcLoc[0][j]])

        #send buffer locations
        for i in range(self.numArms):
            bufferLoc = bufferLocations[i]
            for j in range(4):
                setupChannel.write(1, [bufferLoc[j]])

        #send arm probabilities
        for i in range(self.numArms):
            setupChannel.write(1, [self.probabilities[i]])

        #send probe map
        for i in range(self.totalNeurons):
            setupChannel.write(1, [probeIDMap[i]])

        # #DEBUG
        # for i in range(self.totalNeurons):
        #     setupChannel.write(1, [32+i])


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
