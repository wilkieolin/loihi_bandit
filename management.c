#include <stdlib.h>
#include <string.h>
#include "management.h"
//parameters like NUMARMS are stored here and modified by the host python program
#include "parameters.h"
#include <time.h>
#include <unistd.h>

int probabilities[NUMARMS];
int spike_counts[NUMARMS];
int probe_map[TOTALNEURONS];

int readChannelID = -1;
int writeChannelID = -1;
int rewardChannelID = -1;
int spikeChannelID = -1;

int rewardCompartment[NUMARMS][4];
int punishCompartment[NUMARMS][4];
int voting_epoch = 128;
int cseed = 12340;
int cycle = 0;

int check(runState *s) {
  if (s->time_step == 1) {
    //setup the channels
    readChannelID = getChannelID("setupChannel");
    writeChannelID = getChannelID("dataChannel");
    rewardChannelID = getChannelID("rewardChannel");
    spikeChannelID = getChannelID("spikeChannel");

    //reinforcementChannelID = getChannelID("reinforcementChannel")

    //read out the length of the voting epoch
    readChannel(readChannelID, &voting_epoch, 1);

    //read the random seed
    readChannel(readChannelID, &cseed, 1);
    srand(cseed);

    //read the location of the stub group so we can send events to the input neurons
    for (int i = 0; i < NUMARMS; i++) {
      readChannel(readChannelID, &rewardCompartment[i][0], 4);
      readChannel(readChannelID, &punishCompartment[i][0], 4);
      //DEBUG
      //printf("%d %d %d %d\n", rewardCompartment[i][0], rewardCompartment[i][1], rewardCompartment[i][2], rewardCompartment[i][3]);
    }

    //read out the probabilities of reward for each arm
    readChannel(readChannelID, &probabilities[0], NUMARMS);

    //setup an array to hold the map for probeid <-> neuron
    printf("Probe IDs: ");
    for (int i = 0; i < TOTALNEURONS; i++) {
      readChannel(readChannelID, &probe_map[i], 1);
      printf("%d ", probe_map[i]);
    }
    printf("\n");

    //setup an array to hold the spike counters & init to zero
    for (int i = 0; i < NUMARMS; i++) { spike_counts[i] = 0; }
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

int get_highest(int *counts) {
  // choose the arm with the highest count, randomly breaking ties

  int highest = -1;
  int i_highest = -1;
  int ties = 0;
  int tie_locations[NUMARMS];
  int choice = -1;

  //find the max
  for (int i = 0; i < NUMARMS; i++) {
    if (counts[i] > highest) {
      highest = counts[i];
      i_highest = i;
    }
  }

  //find any values which are tied to it
  for (int i = 0; i < NUMARMS; i++) {
    if (counts[i] == highest) {
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
  //if there are no arms, something hasn't been set up right
  if (NUMARMS == 0) {
    int error = -1;
    writeChannel(writeChannelID, &error, 1);
    return;
  }

  int neuron_count = 0;
  int probe_id = 0;
  //get the spikes counted on each arm since the last epoch & write out by arm
  for (int i = 0; i < TOTALNEURONS; i++) {
    spike_counts[i] = 0;
    //copy each group's spike count from the Lakemont registers
    probe_id = probe_map[i];
    neuron_count = SPIKE_COUNT[(s->time_step-1)&3][probe_id];
    printf("%d ", neuron_count);
    spike_counts[i] += neuron_count;
    //clear the registers
    SPIKE_COUNT[(s->time_step-1)&3][probe_id] = 0;
    writeChannel(spikeChannelID, &spike_counts[i], 1);
  }
  printf("\n");

  //DEBUG
  //int i_highest = get_highest(&spike_counts[0]);
  int i_highest = cycle % NUMARMS;
  //return the arm which we chose to the host
  writeChannel(writeChannelID, &i_highest, 1);

  int reward = get_reward(probabilities[i_highest]);
  writeChannel(rewardChannelID, &reward, 1);

  if (reward) {
    nx_send_discrete_spike(s->time_step, nx_nth_coreid(rewardCompartment[i_highest][2]), rewardCompartment[i_highest][3]);
  } else {
    nx_send_discrete_spike(s->time_step, nx_nth_coreid(punishCompartment[i_highest][2]), punishCompartment[i_highest][3]);
  }

  cycle++;
  return;
}
