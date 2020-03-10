import nxsdk.api.n2a as nx

def create_prototypes(self, vth=255, logicalCoreId=-1):
    prototypes = {}
    prototypes['vth'] = vth
    #setup compartment prototypes
    c_prototypes = {}
    n_prototypes = {}
    s_prototypes = {}

    #Q Neuron
    c_prototypes['somaProto'] = nx.CompartmentPrototype(vThMant=vth,
                                  compartmentCurrentDecay=4095,
                                  compartmentVoltageDecay=0,
                                  logicalCoreId=logicalCoreId
                                  )

    c_prototypes['spkProto'] = nx.CompartmentPrototype(vThMant=vth,
                                     compartmentCurrentDecay=4095,
                                     compartmentVoltageDecay=0,
                                     thresholdBehavior=2,
                                     logicalCoreId=logicalCoreId
                                     )

    c_prototypes['ememProto'] = nx.CompartmentPrototype(vThMant=vth,
                                     #vMaxExp=15,
                                     compartmentCurrentDecay=4095,
                                     compartmentVoltageDecay=0,
                                     thresholdBehavior=3,
                                     logicalCoreId=logicalCoreId
                                     )

    c_prototypes['somaProto'].addDendrite([c_prototypes['spkProto']],
                                        nx.COMPARTMENT_JOIN_OPERATION.OR)

    c_prototypes['spkProto'].addDendrite([c_prototypes['ememProto']],
                                       nx.COMPARTMENT_JOIN_OPERATION.ADD)

    n_prototypes['qProto'] = nx.NeuronPrototype(c_prototypes['somaProto'])

    #S Inverter
    c_prototypes['invProto'] = nx.CompartmentPrototype(vThMant=vth-1,
                                 compartmentCurrentDecay=4095,
                                 compartmentVoltageDecay=0,
                                 thresholdBehavior=0,
                                 functionalState = 2,
                                 logicalCoreId=logicalCoreId
                                 )

    c_prototypes['spkProto'] = nx.CompartmentPrototype(vThMant=vth-1,
                                     biasMant=vth,
                                     biasExp=6,
                                     thresholdBehavior=0,
                                     compartmentCurrentDecay=4095,
                                     compartmentVoltageDecay=0,
                                     functionalState=2,
                                     logicalCoreId=logicalCoreId
                                    )

    c_prototypes['receiverProto'] = nx.CompartmentPrototype(vThMant=vth-1,
                                     compartmentCurrentDecay=4095,
                                     compartmentVoltageDecay=0,
                                      thresholdBehavior=0,
                                      logicalCoreId=logicalCoreId
                                      )

    c_prototypes['invProto'].addDendrite([c_prototypes['receiverProto']],
                                       nx.COMPARTMENT_JOIN_OPERATION.BLOCK)

    n_prototypes['invNeuron'] = nx.NeuronPrototype(c_prototypes['invProto'])

    #AND
    c_prototypes['andProto'] = nx.CompartmentPrototype(vThMant=vth,
                                compartmentCurrentDecay=4095,
                                compartmentVoltageDecay=4095,
                                logicalCoreId=logicalCoreId
                                )

    #Counter (debug)
    v_th_max = 2**17-1
    c_prototypes['counterProto'] = nx.CompartmentPrototype(vThMant=v_th_max,
                                compartmentCurrentDecay=4095,
                                compartmentVoltageDecay=0,
                                logicalCoreId=logicalCoreId
                                )

    #Connections
    s_prototypes['econn'] = nx.ConnectionPrototype(weight=2)
    s_prototypes['iconn'] = nx.ConnectionPrototype(weight=-2)
    s_prototypes['vthconn'] = nx.ConnectionPrototype(weight=-vth)
    s_prototypes['spkconn'] = nx.ConnectionPrototype(weight=vth)
    s_prototypes['halfconn'] = nx.ConnectionPrototype(weight = int(vth/2)+1)
    s_prototypes['thirdconn'] = nx.ConnectionPrototype(weight = int(vth/3)+1)
    s_prototypes['single'] = nx.ConnectionPrototype(weight = 2)


    prototypes['c_prototypes'] = c_prototypes
    prototypes['n_prototypes'] = n_prototypes
    prototypes['s_prototypes'] = s_prototypes

    return prototypes
