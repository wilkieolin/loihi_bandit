import os
import bandit
import prototypes
import nxsdk.api.n2a as nx
import numpy as np
import re
from nxsdk.graph.monitor.probes import *
from nxsdk.graph.processes.phase_enums import Phase


class tracker:
    def __init__(self, network, prototypes, **kwargs):
        self.n_states = n_states
        self.n_per_state = n_per_state
        self.totalNeurons = n_states * n_per_state
        self.l_epoch = l_epoch
        self.epsilon = int(100*epsilon)

        self.recordWeights = kwargs.get('recordWeights', False)
        self.recordSpikes = kwargs.get('recordSpikes', False)

        #initialize the network
        self.net = network
        self.vth = 255
        if prototypes is not None:
            self.c_prototypes = prototypes['c_prototypes']
            self.n_prototypes = prototypes['n_prototypes']
            self.s_prototypes = prototypes['s_prototypes']
        else:
            #setup the necessary NX prototypes
            self._create_prototypes()

        self._create_trackers()

    def _create_prototypes(self):
        prototypes = prototypes.create_prototypes(self.vth)

        self.c_prototypes = prototypes['c_prototypes']
        self.n_prototypes = prototypes['n_prototypes']
        self.s_prototypes = prototypes['s_prototypes']

    def _create_probes(self):

        # -- Create Probes --
        self.probes = {}

        if self.recordSpikes:
            self.probes['spks'] = self.compartments['soma'].probe(nx.ProbeParameter.SPIKE)
            self.probes['nspks'] = self.neurons['invneurons'].soma.probe(nx.ProbeParameter.SPIKE)

            self.probes['eand'] = self.compartments['exc_ands'].probe(nx.ProbeParameter.SPIKE)
            self.probes['iand'] = self.compartments['inh_ands'].probe(nx.ProbeParameter.SPIKE)

        #self.vSpkProbe = self.integrator.probe(nx.ProbeParameter.SPIKE)
        #self.rwdProbe = self.inputs.probe(nx.ProbeParameter.SPIKE)
        if self.recordWeights:
            self.probes['weights'] = self.compartments['memory'].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE)
            self.probes['counters'] = self.compartments['counters'].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE)
            #self.probes['vspks'] = self.compartments['soma'].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE)
            #self.probes['vnspks'] = self.neurons['invneurons'].soma.probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE)

    def _create_trackers(self):
        # -- Create Compartments & Neurons --
        self.compartments = {}
        self.connections = {}
        self.connection_maps = {}
        self.neurons = {}

        c_prototypes = self.c_prototypes
        n_prototypes = self.n_prototypes
        s_prototypes = self.s_prototypes

        #create Q & wire neurons
        qneurons = self.net.createNeuronGroup(size=self.totalNeurons,
                                         prototype=n_prototypes['qProto'])

        qneurons_softreset = qneurons.soma.connect(qneurons.dendrites[0],
                                                  prototype=s_prototypes['vthconn'],
                                                  connectionMask=np.identity(self.totalNeurons))

        memory = qneurons.dendrites[0].dendrites[0]

        self.neurons['qneurons'] = qneurons
        self.connections['qneurons_softreset'] = qneurons_softreset
        self.compartments['soma'] = qneurons.soma
        self.compartments['integrator'] = qneurons.dendrites[0]
        self.compartments['memory'] = memory

        #create & wire inverters
        invneurons = self.net.createNeuronGroup(size=self.totalNeurons,
                                     prototype=n_prototypes['invNeuron'])

        #(Provides the constant 1 signal to inverters)
        driver = self.net.createCompartmentGroup(size=1,
                                            prototype=c_prototypes['spkProto'])

        driver_connection = driver.connect(invneurons.soma,
                                           prototype=s_prototypes['spkconn'],
                                          connectionMask=np.ones((self.totalNeurons,1)))

        self.neurons['invneurons'] = invneurons
        self.compartments['driver'] = driver
        self.connections['driver_connection'] = driver_connection

        #create ANDs
        exc_ands = self.net.createCompartmentGroup(size=self.totalNeurons,
                                         prototype=c_prototypes['andProto'])

        inh_ands = self.net.createCompartmentGroup(size=self.totalNeurons,
                                         prototype=c_prototypes['andProto'])

        self.compartments['exc_ands'] = exc_ands
        self.compartments['inh_ands'] = inh_ands

        #create input stubs for R/P signals to interface with

        estubs = self.net.createInputStubGroup(size=self.numArms)
        istubs = self.net.createInputStubGroup(size=self.numArms)

        self.stubs['estubs'] = estubs
        self.stubs['istubs'] = istubs

        #create the mask that will map the reward/punishment stubs to the right q-trackers,
        # and q-trackers to output
        self.connection_maps['tracker_to_stub'] = np.tile(np.identity(self.n_states), self.n_per_state)
        self.connection_maps['stub_to_tracker'] = tracker_to_stub.transpose()

        # -- Create Higher Connections --
        # Q to inverter
        qinv_conns = qneurons.soma.connect(invneurons.dendrites[0],
                                         prototype=s_prototypes['spkconn'],
                                          connectionMask=np.identity(self.totalNeurons))
        # &EXC to Q memory
        exc_conns = exc_ands.connect(memory,
                                    prototype=s_prototypes['econn'],
                                    connectionMask=np.identity(self.totalNeurons))

        # &INH to Q memory
        inh_conns = inh_ands.connect(memory,
                                    prototype=s_prototypes['iconn'],
                                    connectionMask=np.identity(self.totalNeurons))

        # !Q to &EXC
        nspk_exc_conns = invneurons.soma.connect(exc_ands,
                                                prototype=s_prototypes['halfconn'],
                                                connectionMask=np.identity(self.totalNeurons))

        # Q to &INH
        spk_inh_conns = qneurons.soma.connect(inh_ands,
                                             prototype=s_prototypes['halfconn'],
                                             connectionMask=np.identity(self.totalNeurons))

        # Exc stub to &EXC
        estub_inh_conn = estubs.connect(inh_ands,
                                        prototype=s_prototypes['halfconn'],
                                        connectionMask=stub_to_tracker)

        # Inh stub to &INH
        istub_exc_conn = istubs.connect(exc_ands,
                                        prototype=s_prototypes['halfconn'],
                                        connectionMask=stub_to_tracker)

        self.connections['qinv_conns'] = qinv_conns
        self.connections['exc_conns'] = exc_conns
        self.connections['inh_conns'] = inh_conns
        self.connections['nspk_exc_conns'] = nspk_exc_conns
        self.connections['spk_inh_conns'] = spk_inh_conns
        self.connections['estub_exc_conn'] = estub_inh_conn
        self.connections['istub_inh_conn'] = istub_exc_conn

        #counter
        counters = self.net.createCompartmentGroup(size=self.n_states,
                                         prototype=self.c_prototypes['counterProto'])

        self.connections['soma_counter'] = invneurons.soma.connect(counters,
                                         prototype=self.s_prototypes['single'],
                                         connectionMask=self.connection_maps['tracker_to_stub'])

        self.compartments['counters'] = counters

    def get_counter_locations(self):
        locs = []
        for i in range(self.n_states):
            compartmentId = self.compartments['counters'][i].nodeId
            compartmentLoc = self.net.resourceMap.compartmentMap[compartmentId]

            locs.append(compartmentLoc)

        return locs

    def get_probeid_map(self):
        assert self.recordSpikes, "Must be recording spikes to have a probemap to access."
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
