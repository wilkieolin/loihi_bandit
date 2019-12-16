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
        self._create_trackers()
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

        # need to send the epoch length, seed, probability (numArms), and pos/neg terminal for each arm,
        # and the probeID <-> neuron map
        n_outData = 2 + self.numArms * (1 + 2*4) + self.totalNeurons
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
        c_prototypes = {}
        n_prototypes = {}
        s_prototypes = {}

        #Q Neuron
        c_prototypes['somaProto'] = nx.CompartmentPrototype(vThMant=self.vth,
                                        compartmentCurrentDecay=4095,
                                        compartmentVoltageDecay=0)

        c_prototypes['spkProto'] = nx.CompartmentPrototype(vThMant=self.vth,
                                           compartmentCurrentDecay=4095,
                                           compartmentVoltageDecay=0,
                                           thresholdBehavior=2)

        c_prototypes['ememProto'] = nx.CompartmentPrototype(vThMant=self.vth,
                                           #vMaxExp=15,
                                           compartmentCurrentDecay=4095,
                                           compartmentVoltageDecay=0,
                                           thresholdBehavior=3)

        c_prototypes['somaProto'].addDendrite([c_prototypes['spkProto']],
                                              nx.COMPARTMENT_JOIN_OPERATION.OR)

        c_prototypes['spkProto'].addDendrite([c_prototypes['ememProto']],
                                             nx.COMPARTMENT_JOIN_OPERATION.ADD)

        n_prototypes['qProto'] = nx.NeuronPrototype(c_prototypes['somaProto'])

        #S Inverter
        c_prototypes['invProto'] = nx.CompartmentPrototype(vThMant=self.vth-1,
                                       compartmentCurrentDecay=4095,
                                       compartmentVoltageDecay=0,
                                       thresholdBehavior=0,
                                       functionalState = 2
                                       )

        c_prototypes['spkProto'] = nx.CompartmentPrototype(vThMant=self.vth-1,
                                           biasMant=self.vth,
                                           biasExp=6,
                                           thresholdBehavior=0,
                                           compartmentCurrentDecay=4095,
                                           compartmentVoltageDecay=0,
                                           functionalState=2
                                          )

        c_prototypes['receiverProto'] = nx.CompartmentPrototype(vThMant=self.vth-1,
                                           compartmentCurrentDecay=4095,
                                           compartmentVoltageDecay=0,
                                            thresholdBehavior=0)

        c_prototypes['invProto'].addDendrite([c_prototypes['receiverProto']],
                                             nx.COMPARTMENT_JOIN_OPERATION.BLOCK)

        n_prototypes['invNeuron'] = nx.NeuronPrototype(c_prototypes['invProto'])

        #AND
        c_prototypes['andProto'] = nx.CompartmentPrototype(vThMant=self.vth,
                                      compartmentCurrentDecay=4095,
                                      compartmentVoltageDecay=4095)

        #Connections
        s_prototypes['econn'] = nx.ConnectionPrototype(weight=2)
        s_prototypes['iconn'] = nx.ConnectionPrototype(weight=-2)
        s_prototypes['vthconn'] = nx.ConnectionPrototype(weight=-self.vth)
        s_prototypes['spkconn'] = nx.ConnectionPrototype(weight=self.vth)
        s_prototypes['halfconn'] = nx.ConnectionPrototype(weight = int(self.vth/2)+1)


        self.c_prototypes = c_prototypes
        self.n_prototypes = n_prototypes
        self.s_prototypes = s_prototypes

    def _create_probes(self):

        # -- Create Probes --
        self.probes = {}

        customSpikeProbeCond = SpikeProbeCondition(tStart=100000000)
        self.probes['spks'] = self.compartments['soma'].probe(nx.ProbeParameter.SPIKE, customSpikeProbeCond)
        self.probes['nspks'] = self.neurons['invneurons'].soma.probe(nx.ProbeParameter.SPIKE)

        self.probes['eand'] = self.compartments['exc_ands'].probe(nx.ProbeParameter.SPIKE)
        self.probes['iand'] = self.compartments['inh_ands'].probe(nx.ProbeParameter.SPIKE)



        #self.vSpkProbe = self.integrator.probe(nx.ProbeParameter.SPIKE)
        #self.rwdProbe = self.inputs.probe(nx.ProbeParameter.SPIKE)
        if self.recordWeights:
            self.probes['weights'] = self.compartments['memory'].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE)
            self.probes['vspks'] = self.compartments['soma'].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE)
            self.probes['vnspks'] = self.neurons['invneurons'].soma.probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE)

    def _create_trackers(self):
        # -- Create Compartments & Neurons --
        self.compartments = {}
        self.connections = {}
        self.neurons = {}
        self.stubs = {}

        c_prototypes = self.c_prototypes
        n_prototypes = self.n_prototypes
        s_prototypes = self.s_prototypes

        #create Q & wire neurons
        qneurons = self.net.createNeuronGroup(size=self.numArms,
                                         prototype=n_prototypes['qProto'])

        qneurons_softreset = qneurons.soma.connect(qneurons.dendrites[0],
                                                  prototype=s_prototypes['vthconn'],
                                                  connectionMask=np.identity(self.numArms))

        memory = qneurons.dendrites[0].dendrites[0]

        self.neurons['qneurons'] = qneurons
        self.connections['qneurons_softreset'] = qneurons_softreset
        self.compartments['soma'] = qneurons.soma
        self.compartments['integrator'] = qneurons.dendrites[0]
        self.compartments['memory'] = memory

        #create & wire inverters
        invneurons = self.net.createNeuronGroup(size=self.numArms,
                                     prototype=n_prototypes['invNeuron'])

        driver = self.net.createCompartmentGroup(size=1,
                                            prototype=c_prototypes['spkProto'])

        driver_connection = driver.connect(invneurons.soma,
                                           prototype=s_prototypes['spkconn'],
                                          connectionMask=np.ones((self.numArms,1)))

        self.neurons['invneurons'] = invneurons
        self.compartments['driver'] = driver
        self.connections['driver_connection'] = driver_connection

        #create ANDs
        exc_ands = self.net.createCompartmentGroup(size=self.numArms,
                                         prototype=c_prototypes['andProto'])

        inh_ands = self.net.createCompartmentGroup(size=self.numArms,
                                         prototype=c_prototypes['andProto'])

        self.compartments['exc_ands'] = exc_ands
        self.compartments['inh_ands'] = inh_ands

        #create input stubs for SNIP to interface with the network

        estubs = self.net.createInputStubGroup(size=self.numArms)
        istubs = self.net.createInputStubGroup(size=self.numArms)

        self.stubs['estubs'] = estubs
        self.stubs['istubs'] = istubs

        # -- Create Higher Connections --
        # Q to inverter
        qinv_conns = qneurons.soma.connect(invneurons.dendrites[0],
                                         prototype=s_prototypes['spkconn'],
                                          connectionMask=np.identity(self.numArms))
        # &EXC to Q memory
        exc_conns = exc_ands.connect(memory,
                                    prototype=s_prototypes['econn'],
                                    connectionMask=np.identity(self.numArms))

        # &INH to Q memory
        inh_conns = inh_ands.connect(memory,
                                    prototype=s_prototypes['iconn'],
                                    connectionMask=np.identity(self.numArms))

        # !Q to &EXC
        nspk_exc_conns = invneurons.soma.connect(exc_ands,
                                                prototype=s_prototypes['halfconn'],
                                                connectionMask=np.identity(self.numArms))

        # Q to &INH
        spk_inh_conns = qneurons.soma.connect(inh_ands,
                                             prototype=s_prototypes['halfconn'],
                                             connectionMask=np.identity(self.numArms))

        # Exc stub to &EXC
        estub_exc_conn = estubs.connect(exc_ands,
                                        prototype=s_prototypes['halfconn'],
                                        connectionMask=np.identity(self.numArms))

        istub_inh_conn = istubs.connect(inh_ands,
                                        prototype=s_prototypes['halfconn'],
                                        connectionMask=np.identity(self.numArms))

        self.connections['qinv_conns'] = qinv_conns
        self.connections['exc_conns'] = exc_conns
        self.connections['inh_conns'] = inh_conns
        self.connections['nspk_exc_conns'] = nspk_exc_conns
        self.connections['spk_inh_conns'] = spk_inh_conns
        self.connections['estub_exc_conn'] = estub_exc_conn
        self.connections['istub_inh_conn'] = istub_inh_conn


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
            eAxonId = self.connections['estub_exc_conn'][i].inputAxon.nodeId
            eAxon = self.net.resourceMap.inputAxon(eAxonId)[0]

            iAxonId = self.connections['istub_inh_conn'][i].inputAxon.nodeId
            iAxon = self.net.resourceMap.inputAxon(iAxonId)[0]

            locs.append((eAxon, iAxon))

        return locs

    def get_mean_optimal_action(self):
        assert hasattr(self, 'choices'), "Must run network to get MOA."
        bestarm = np.argmax(self.probabilities)
        #TODO finish

    def get_probeid_map(self):
        pids = [self.probes['spks'][0][i].n2Probe.counterId for i in range(self.totalNeurons)]

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
            setupChannel.write(4, bufferLoc[0])
            setupChannel.write(4, bufferLoc[1])

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
