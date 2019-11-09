import os
import nxsdk.api.n2a as nx
import numpy as np
import matplotlib.pyplot as plt
import re
from nxsdk.graph.monitor.probes import *
from nxsdk.graph.processes.phase_enums import Phase

def create_network(numArms, neuronsPerArm, epochs, weights, probabilities):
    assert len(weights) == numArms, "Must provide number of weights equal to number of arms."
    assert len(probabilities) == numArms, "Must provide probability for each arm's reward."
    for p in probabilities:
        assert p in range(0,100), "Probabilities must be represented as int from 0-100."
    totalNeurons = neuronsPerArm * numArms

    #create the network
    net = nx.NxNet()
    #set up the noisy source neuron prototype
    p_2Fire = nx.CompartmentPrototype(biasMant=0,
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
    #set up the integrative prototype
    p_Compare = nx.CompartmentPrototype(biasMant=0,
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

    #set up the learning rule that will control weight changes
    lr = net.createLearningRule(dw='u0*r1',
                                r1Impulse=1,
                                r1TimeConstant=1,
                                tEpoch=32)
                                #TODO - can inc/dec weight based on reward

    #create the connections which drive the integrator
    exh_connx = nx.ConnectionPrototype(weight=2,
                                    delay=0,
                                    enableLearning=1,
                                    learningRule=lr,
                                    signMode=nx.SYNAPSE_SIGN_MODE.MIXED)
    inh_connx = nx.ConnectionPrototype(weight=2,
                                    delay=0,
                                    signMode=nx.SYNAPSE_SIGN_MODE.MIXED)

    compartmentGroups = []
    synapseGroups = []
    monitors = []

    for arm in range(numArms):
        w = weights[arm]

        #create inhibitory, excitatory, and integrative compartment groups
        inhDriver = net.createCompartmentGroup(size=neuronsPerArm, prototype=p_2Fire)
        exhDriver = net.createCompartmentGroup(size=neuronsPerArm, prototype=p_2Fire)
        comparator = net.createCompartmentGroup(size=neuronsPerArm, prototype=p_Compare)
        compartmentGroups.append(inhDriver)
        compartmentGroups.append(exhDriver)
        compartmentGroups.append(comparator)

        exhGrp = exhDriver.connect(comparator,
                      prototype=exh_connx,
                      weight= np.repeat(w, neuronsPerArm) * np.identity(neuronsPerArm),
                     connectionMask=np.identity(neuronsPerArm))

        inhGrp = inhDriver.connect(comparator,
                      prototype=inh_connx,
                      weight=-50*np.identity(neuronsPerArm),
                     connectionMask=np.identity(neuronsPerArm))

        synapseGroups.append(exhGrp)
        synapseGroups.append(inhGrp)
        #setup the spike monitor
        customSpikeProbeCond = SpikeProbeCondition(tStart=10000000)
        monitors.append(comparator.probe(nx.ProbeParameter.SPIKE, customSpikeProbeCond))

    #compile the network so we can add channels for the SNIPs
    compiler = nx.N2Compiler()
    board = compiler.compile(net)
    #get the location of the reinforcementChannel
    rc_loc = net.resourceMap.reinforcementChannel(0)

    #setup the management SNIP to calculate rewards and choose numArms
    includeDir = os.getcwd()
    learning = board.createSnip(Phase.EMBEDDED_MGMT,
                                 includeDir=includeDir,
                                 cFilePath = includeDir + "/management.c",
                                 funcName = "run_cycle",
                                 guardName = "check")

    #create a channel to communicate with the Lakemont what the probability of each arm is
    #have to transfer the number of arms, neurons per arm, and the probeID listening to each neuron
    setupChannel = board.createChannel(b'setupChannel', "int", numArms + 4 + totalNeurons)
    setupChannel.connect(None, learning)
    #setup the channels which will transfer the chosen arm
    dataChannel = board.createChannel(b'dataChannel', "int", (epochs+1))
    dataChannel.connect(learning, None)
    #and the reward for that arm
    rewardChannel = board.createChannel(b'rewardChannel', "int", (epochs+1))
    rewardChannel.connect(learning, None)

    #write the parameters into the file since we can't use dynamic allocation ;_;
    set_params_file(numArms, neuronsPerArm)

    #boot the board
    board.startDriver()
    #set the reward probabilities
    setupChannel.write(numArms, probabilities)
    #write the location of the reinforcementChannel we can send reward events to
    for i in range(4):
        setupChannel.write(1, [rc_loc[0][i]])
    #and the map for probeid <-> neuron
    pids = [[monitors[j][0].probes[i].n2Probe.counterId for i in range(neuronsPerArm)] for j in range(numArms)]
    for i in range(numArms):
        for j in range(neuronsPerArm):
            setupChannel.write(1, [pids[i][j]])

    return (board, monitors, dataChannel, rewardChannel)

def set_params_file(numArms, neuronsPerArm):
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
