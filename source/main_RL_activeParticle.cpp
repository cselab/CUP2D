#include <unistd.h>
#include <sys/stat.h>

#include "smarties.h"
#include "Simulation.h"
#include "Obstacles/activeParticle.h"

using namespace cubism;

// Implement particle swarm later (shield etc.)

static bool bVerbose = true;

inline void resetIC(activeParticle* const agent, smarties::Communicator*const comm)
{ 
  const Real initialOuterRadius = 0.95;
  const Real initialInnerRadius = 0.05;
  const Real theta_0_range = M_PI;

  std::uniform_real_distribution<Real> disRad(initialInnerRadius, initialOuterRadius);
  std::uniform_real_distribution<Real> disAng(-theta_0_range, theta_0_range);
  double initialRadius =  comm->isTraining() ? disRad(comm->getPRNG()) : 0.1;
  double initialAngle = comm->isTraining() ? disAng(comm->getPRNG())  : 0;  
  
  double C[2] = {initialRadius*std::cos(initialAngle), initialRadius*std::sin(initialAngle)};
  agent->setCenterOfMass(C);
}

inline void setAction(activeParticle* const agent, const std::vector<double> act, double t)
{ 
  bool actUACM = act[0] > 0.5 ? true : false;
  if(actUACM){
    agent->finalAngRotation = act[1]*agent->forcedOmegaCirc;
    agent->tStartAccelTransfer = t;
  }
  else{
    agent->finalRadiusRotation = act[2]*agent->forcedRadiusMotion;
    agent->tStartElliTransfer = t;
  }
}

inline std::vector<double> getState(const activeParticle* const agent)
{
  double currentRadius = agent->forcedRadiusMotion;
  double currentOmegaCirc = agent->forcedOmegaCirc;
  std::vector<double> state = {currentRadius, currentOmegaCirc};
  if(bVerbose){
    printf("Sending [%f %f]\n", currentRadius, currentOmegaCirc);
    fflush(0);
  }
  return state;
}

inline double getReward(activeParticle* const agent)
{
  double reward = agent->reward();

  return reward;
}

inline bool isTerminal(const activeParticle* const agent)
{
  if(agent->finalRadiusRotation > 0.95) return true;
  if(agent->finalRadiusRotation < 0.05) return true;
  if(agent->forcedOmegaCirc >= 80) return true;

  return false;
}

inline bool checkNaN(std::vector<double>& state, double& reward)
{
  bool bTrouble = false;
  if(std::isnan(reward)) bTrouble = true;
  for(size_t i=0; i<state.size(); i++) if(std::isnan(state[i])) bTrouble = true;
  if(bTrouble) {
    reward = -100;
    printf("Caught a NaN!\n");
    state = std::vector<double>(state.size(), 0);
  }
  return bTrouble;
}

inline double getTimeToNextAct(const activeParticle* const agent, double t)
{
  if(agent->lastUCM) return t + 2*M_PI/agent->forcedOmegaCirc;
  if(agent->lastUACM) return t + agent->tTransitAccel;
  if(agent->lastEM) return t + agent->tTransitElli;

  return 1;
}

inline void app_main(
  smarties::Communicator*const comm,  // communicator with smarties
  MPI_Comm mpicom,                    // mpi_comm that mpi-based apps can use
  int argc, char**argv                // args read from app's runtime settings file
) {
  const int nActions = 3, nStates = 2;
  const unsigned maxLearnStepPerSim = comm->isTraining() ? 200
                                     : std::numeric_limits<int>::max();

  comm->setStateActionDims(nStates, nActions);

  Simulation sim(argc, argv);
  sim.init();
  
  activeParticle* const agent = dynamic_cast<activeParticle*>( sim.getShapes()[0] );
  if(agent==nullptr) { printf("Obstacle was not an Active Particle!\n"); abort(); }
  if(comm->isTraining() == false) {
    sim.sim.verbose = true; sim.sim.muteAll = false;
    sim.sim.dumpTime = 0.3;
  }
  unsigned sim_id = 0, tot_steps = 0;

  // Terminate loop if reached max number of time steps. Never terminate if 0
  while( 1 ) // train loop
  {
    if(comm->isTraining() == false)
    {
      char dirname[1024]; dirname[1023] = '\0';
      sprintf(dirname, "run_%08u/", sim_id);
      printf("Starting a new sim in directory %s\n", dirname);
      mkdir(dirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      chdir(dirname);
    }

    sim.reset();
    resetIC(agent, comm); // randomize initial conditions

    double t = 0;
    unsigned step = 0;
    bool agentOver = false;

    comm->sendInitState(getState(agent)); //send initial state

    while (true) //simulation loop
    {
      setAction(agent, comm->recvAction(), t);
      const double tNextAct = getTimeToNextAct(agent, t);
      while (t < tNextAct)
      {
        const double dt = sim.calcMaxTimestep();
        t += dt;

        if ( sim.advance( dt ) ) { // if true sim has ended
          printf("Set -tend 0. This file decides the length of train sim.\n");
          assert(false); fflush(0); abort();
        }
        if ( isTerminal(agent) ) {
          agentOver = true;
          break;
        }
      }
      step++;
      tot_steps++;
      std::vector<double> state = getState(agent);
      double reward = getReward(agent);

      if(agentOver||checkNaN(state, reward)){
        printf("Agent failed after %u steps.\n", step); fflush(0);
        comm->sendTermState(state, reward);
        break;
      }
      else
      if(step >= maxLearnStepPerSim) {
        printf("Sim ended\n"); fflush(0);
        comm->sendLastState(state, reward);
        break;
      }
      else comm->sendState(state, reward);
    } // simulation is done

    if(comm->isTraining()==false) chdir("../");
    sim_id++;

    if(comm->terminateTraining()) return; // exit program
  }
}

int main(int argc, char**argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  e.run( app_main );
  return 0;
}
