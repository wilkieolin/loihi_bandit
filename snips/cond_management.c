#include <stdlib.h>
#include <string.h>
#include "cond_management.h"
//parameters like N_STATES are stored here and modified by the host python program
#include "cond_parameters.h"
#include <time.h>
#include <unistd.h>

int probabilities[N_CONDITIONS][N_STATES];
int counterVoltages[N_CONDITIONS][N_STATES];

int readChannelID = -1;
int writeChannelID = -1;
int rewardChannelID = -1;
int spikeChannelID = -1;

int rewardCompartment[4];
int punishCompartment[4];
int stateCompartments[N_STATES];
int conditionCompartments[N_CONDITIONS];
int counterCompartment[N_CONDITIONS][N_STATES][4];

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
    readChannel(readChannelID, &probabilities[0], N_ESTIMATES);

    printf("Got variables\n");
    //read the location of the stub group so we can send events to the reward/punishment stubs
    for (int i = 0; i < N_STATES; i++) {
      readChannel(readChannelID, &rewardCompartment[i][0], 4);
      readChannel(readChannelID, &punishCompartment[i][0], 4);
      //DEBUG
      //printf("%d %d %d %d\n", rewardCompartment[i][0], rewardCompartment[i][1], rewardCompartment[i][2], rewardCompartment[i][3]);
    }
    printf("Got R/P compartments\n");

    //read the location of the counter neurons
    for (int i = 0; i < N_ESTIMATES; i++) {
      readChannel(readChannelID, &counterCompartment[i][0], 4);
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

void get_counter_voltages(int condition) {
  CoreId core;
  int cxId = 0;
  NeuronCore *nc;
  CxState cxs;

  //read out the counter soma voltages
  //printf("Voltages: ");
  for (int i = 0; i < N_STATES; i++) {
    //get the core the counter is on
    core = nx_nth_coreid(counterCompartment[state][i][2]);
    nc = NEURON_PTR(core);
    //get the compartment the voltage is in
    cxId = counterCompartment[state][i][3];
    cxs = nc->cx_state[cxId];
    counterVoltages[i] = cxs.V;
    nc->cx_state[cxId].V = 0;
    //printf("%d ", counterVoltages[i]);
  }
  //printf("\n");

  return;
}

void reset_counter_voltages() {
  CoreId core;
  int cxId = 0;
  NeuronCore *nc;
  CxState cxs;

  for (int i = 0; i < N_CONDITIONS; i++) {
    for (int j = 0; j < N_STATES; j++) {
      //get the core the counter is on
      core = nx_nth_coreid(counterCompartment[i][j][2]);
      nc = NEURON_PTR(core);
      //get the compartment the voltage is in
      cxId = counterCompartment[i][j][3];
      cxs = nc->cx_state[cxId];
      //reset it to zero
      nc->cx_state[cxId].V = 0;
    }
  }

  return;
}

int get_highest() {
  // choose the arm with the highest count, randomly breaking ties

  int highest = -1;
  int i_highest = -1;
  int ties = 0;
  int tie_locations[N_STATES];
  int choice = -1;

  //find the max
  for (int i = 0; i < N_STATES; i++) {
    if (counterVoltages[i] > highest) {
      highest = counterVoltages[i];
      i_highest = i;
    }
  }

  //find any values which are tied to it
  for (int i = 0; i < N_STATES; i++) {
    if (counterVoltages[i] == highest) {
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
  if (N_STATES == 0 || N_CONDITIONS == 0) {
    int error = -1;
    writeChannel(writeChannelID, &error, 1);
    return;
  }

  //select the condition we're going to be sampling under
  int condition = rand() % N_CONDITIONS;

  //get the firing rates for the neurons representing our condition
  get_counter_voltages(condition);
  reset_counter_voltages();

  for (int i = 0; i < N_STATES; i++) {
    writeChannel(spikeChannelID, &counterVoltages[i], 1);
  }

  int i_highest = -1;
  if (rand() % 100 < epsilon) {
    //choose a random arm with p = eps.
    i_highest = rand() % N_STATES;
  } else {
    //choose the best arm with p = (1-eps.)
    i_highest = get_highest();
  }

  //return the arm which we chose to the host and the condition
  writeChannel(writeChannelID, &i_highest, 1);
  writeChannel(writeChannelID, &condition, 1);

  int reward = get_reward(probabilities[condition][i_highest]);
  writeChannel(rewardChannelID, &reward, 1);

  if (reward) {
    nx_send_discrete_spike(s->time_step, nx_nth_coreid(rewardCompartment[i_highest][2]), rewardCompartment[i_highest][3]);
  } else {
    nx_send_discrete_spike(s->time_step, nx_nth_coreid(punishCompartment[i_highest][2]), punishCompartment[i_highest][3]);
  }

  cycle++;
  return;
}
