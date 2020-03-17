#include <stdlib.h>
#include <string.h>
#include "cond_management.h"
//parameters like N_ACTIONS are stored here and modified by the host python program
#include "cond_parameters.h"
#include <time.h>
#include <unistd.h>

int probabilities[N_STATES][N_ACTIONS];
int counterVoltages[N_STATES][N_ACTIONS];

int readChannelID = -1;
int writeChannelID = -1;
int rewardChannelID = -1;
int spikeChannelID = -1;

int rewardCompartment[4];
int punishCompartment[4];
int stateCompartments[N_ACTIONS][4];
int conditionCompartments[N_STATES][4];
int counterCompartment[N_STATES][N_ACTIONS][4];

int voting_epoch = 128;
int epsilon = 10;
int cseed = 12340;
int cycle = 0;

int check(runState *s) {
  if (s->time_step == 1) {
    printf("Setting up...\n");
    //setup the channels
    readChannelID = getChannelID("setupChannel");
    writeChannelID = getChannelID("dataChannel");
    rewardChannelID = getChannelID("rewardChannel");
    spikeChannelID = getChannelID("spikeChannel");

    //reinforcementChannelID = getChannelID("reinforcementChannel")

    //read out the length of the voting epoch
    readChannel(readChannelID, &voting_epoch, 1);

    //read the epsilon
    readChannel(readChannelID, &epsilon, 1);

    //read the random seed
    readChannel(readChannelID, &cseed, 1);
    srand(cseed);

    //read out the probabilities of reward for each arm
    for (int i = 0; i < N_STATES; i++) {
      for (int j = 0; j < N_ACTIONS; j++) {
        readChannel(readChannelID, &probabilities[i][j], 1);
      }
    }

    printf("Got variables\n");
    //read the location of the stub group so we can send events to the reward/punishment stubs
    readChannel(readChannelID, &rewardCompartment[0], 4);
    readChannel(readChannelID, &punishCompartment[0], 4);

    //read the location of the state stubs
    for (int i = 0; i < N_ACTIONS; i++) {
      readChannel(readChannelID, &stateCompartments[i][0], 4);
    }

    //read the location of the condition stubs
    for (int i = 0; i < N_STATES; i++) {
      readChannel(readChannelID, &conditionCompartments[i][0], 4);
    }
    printf("Got R/P/State/Condition compartments\n");

    //read the location of the counter neurons
    for (int i = 0; i < N_STATES; i++) {
      for (int j = 0; j < N_ACTIONS; j++) {
        readChannel(readChannelID, &counterCompartment[i][j][0], 4);
      }
    }
    printf("Got Counter compartments, done.\n");
  }

  if (s->time_step % voting_epoch == 0) {
    return 1;
  } else {
    return 0;
  }
}

int get_reward(int p) {
  int r = rand() % 100;
  if (r < p) {
    return 1;
  } else {
    return 0;
  }
}

void get_counter_voltages() {
  int cxId = 0;

  CoreId core;
  NeuronCore *nc;
  CxState cxs;


  //read out the counter soma voltages
  //printf("Voltages: ");
  for (int i = 0; i < N_STATES; i++) {
    for (int j = 0; j < N_ACTIONS; j++) {
      //get the core the counter is on
      core = nx_nth_coreid(counterCompartment[i][j][2]);
      nc = NEURON_PTR(core);
      //get the compartment the voltage is in
      cxId = counterCompartment[i][j][3];
      cxs = nc->cx_state[cxId];
      counterVoltages[i][j] = cxs.V;
    }
  }
  //printf("\n");

  return;
}

void reset_counter_voltages() {
  CoreId core;
  int cxId = 0;
  NeuronCore *nc;

  for (int i = 0; i < N_STATES; i++) {
    for (int j = 0; j < N_ACTIONS; j++) {
      //get the core the counter is on
      core = nx_nth_coreid(counterCompartment[i][j][2]);
      nc = NEURON_PTR(core);
      //get the compartment the voltage is in
      cxId = counterCompartment[i][j][3];
      //reset it to zero
      nc->cx_state[cxId].V = 0;
    }
  }

  return;
}

int get_highest(int condition) {
  // choose the arm with the highest count, randomly breaking ties

  int highest = -1;
  int i_highest = -1;
  int ties = 0;
  int tie_locations[N_ACTIONS];
  int choice = -1;

  //find the max
  for (int i = 0; i < N_ACTIONS; i++) {
    if (counterVoltages[condition][i] > highest) {
      highest = counterVoltages[condition][i];
      i_highest = i;
    }
  }

  //find any values which are tied to it
  for (int i = 0; i < N_ACTIONS; i++) {
    if (counterVoltages[condition][i] == highest) {
      ties++;
      tie_locations[i] = 1;
    } else {
      tie_locations[i] = 0;
    }
  }

  //choose randomly among ties if necessary
  if (ties > 1) {
    i_highest = rand() % ties + 1;

    int count = 0;
    int i = 0;
    while (count != i_highest) {
      count += tie_locations[i];
      i++;
    }
    choice = i - 1;

  } else {
    choice = i_highest;
  }

  return choice;
}


void run_cycle(runState *s) {
  //if there are no states or conditions
  if (N_ACTIONS == 0 || N_STATES == 0) {
    int error = -1;
    writeChannel(writeChannelID, &error, 1);
    return;
  }

  //select the condition we're going to be sampling under
  int condition = rand() % N_STATES;

  //get the firing rates for all estimate neurons
  get_counter_voltages();
  reset_counter_voltages();

  for (int i = 0; i < N_STATES; i++) {
    for (int j = 0; j < N_ACTIONS; j++) {
      writeChannel(spikeChannelID, &counterVoltages[i][j], 1);
    }
  }

  int i_highest = -1;
  if (rand() % 100 < epsilon) {
    //choose a random arm with p = eps.
    i_highest = rand() % N_ACTIONS;
  } else {
    //choose the best arm with p = (1-eps.)
    i_highest = get_highest(condition);
  }

  //return the arm which we chose to the host and the condition
  writeChannel(writeChannelID, &i_highest, 1);
  writeChannel(writeChannelID, &condition, 1);

  int reward = get_reward(probabilities[condition][i_highest]);
  writeChannel(rewardChannelID, &reward, 1);


  //identify state
  nx_send_discrete_spike(s->time_step, nx_nth_coreid(stateCompartments[i_highest][2]), stateCompartments[i_highest][3]);
  //identify condition
  nx_send_discrete_spike(s->time_step, nx_nth_coreid(conditionCompartments[condition][2]), conditionCompartments[condition][3]);
  //identify reward/punishment
  if (reward) {
    nx_send_discrete_spike(s->time_step, nx_nth_coreid(rewardCompartment[2]), rewardCompartment[3]);
  } else {
    nx_send_discrete_spike(s->time_step, nx_nth_coreid(punishCompartment[2]), punishCompartment[3]);
  }

  cycle++;
  return;
}
