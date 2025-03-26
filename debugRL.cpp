

#include "Obstacles/StefanFish.h"
#include "Simulation.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#define NACTIONS 2

std::vector<std::vector<Real>> readActions(const int Nagents);

int main(int argc, char **argv) {
  int threadSafety;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSafety);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Simulation *_environment = new Simulation(argc, argv, MPI_COMM_WORLD);
  _environment->init();

  const int Nagents = _environment->getShapes().size();
  std::vector<std::vector<Real>> actions;
  int numActions;
  if (rank == 0) {
    actions = readActions(Nagents);
    numActions = actions[0].size() / (NACTIONS + 1);
  }
  MPI_Bcast(&numActions, 1, MPI_INT, 0, MPI_COMM_WORLD);

  Real t = 0;
  Real dtAct;
  Real tNextAct = 0;

  for (int a = 0; a < numActions; a++) {
    for (int i = 0; i < Nagents; i++) {
      StefanFish *agent =
          dynamic_cast<StefanFish *>(_environment->getShapes()[i].get());

      std::vector<Real> action(NACTIONS);
      if (rank == 0)
        for (int j = 1; j < NACTIONS + 1; j++)
          action[j - 1] = (actions[i][a * (NACTIONS + 1) + j]);
      MPI_Bcast(action.data(), NACTIONS, MPI_Real, 0, MPI_COMM_WORLD);
      agent->act(t, action);
    }
    StefanFish *agent0 =
        dynamic_cast<StefanFish *>(_environment->getShapes()[0].get());
    dtAct = agent0->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while (t < tNextAct) {
      const Real dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;
      _environment->advance(dt);
    }
  }
  delete _environment;
  MPI_Finalize();
}

std::vector<std::vector<Real>> readActions(const int Nagents) {
  std::vector<std::vector<Real>> actions(Nagents);
  for (int i = 0; i < Nagents; i++) {
    std::fstream myfile("actions" + std::to_string(i) + ".txt",
                        std::ios_base::in);
    Real a;
    while (myfile >> a)
      actions[i].push_back(a);
  }
  return actions;
}
