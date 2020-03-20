/* main_RL_cStart.cpp
 * Created by Ioannis Mandralis (ioannima@ethz.ch)
 * Main script for a single StefanFish learning a fast start
*/

#include <unistd.h>
#include <sys/stat.h>

#include "smarties.h"
#include "Simulation.h"
#include "Obstacles/StefanFish.h"

using namespace cubism;
//
// All these functions are defined here and not in object itself because
// Many different tasks, each requiring different state/act/rew descriptors
// could be designed for the same objects (also to fully encapsulate RL).
//
// main hyperparameters:
// number of actions per characteristic time scale
// max number of actions per simulation
// range of angles in initial conditions

inline void resetIC(StefanFish* const a, smarties::Communicator*const c)
{
    std::uniform_real_distribution<double> disA(-20./180.*M_PI, 20./180.*M_PI);
    const double SA = c->isTraining()? disA(c->getPRNG()) : 0.00;
    a->setOrientation(SA);
}

inline void setAction(StefanFish* const agent,
                      const std::vector<double> act, const double t)
{
    agent->act(t, act);
}

inline bool isTerminal(const StefanFish*const a) {

    // Terminate when the fish exits a radius of one characteristic lengths
    double charLength = a->getCharLength();
    double com[2] = {0, 0};
    a->getLabPosition(com);
    double radialPos = std::sqrt(std::pow(com[0], 2) + std::pow(com[1], 2));
    return (radialPos >= charLength);
}

inline double getReward(const StefanFish* const a, const double& dt) {
    // Reward is negative the time between each action
    // time dependent
    // distance from initial condition.
    // terminal reward: total time it took
    // minimize energy ? would use the motion with maximum energy...
    // (efficiency)
    // only control the curvature
    return -dt;
}

inline bool checkNaN(std::vector<double>& state, double& reward)
{
    bool bTrouble = false;
    if(std::isnan(reward)) bTrouble = true;
    for(size_t i=0; i<state.size(); i++) if(std::isnan(state[i])) bTrouble = true;
    if ( bTrouble ) {
        reward = -100;
        printf("Caught a nan!\n");
        state = std::vector<double>(state.size(), 0);
    }
    return bTrouble;
}

inline double getTimeToNextAct(const StefanFish* const agent, const double t) {
    return t + agent->getLearnTPeriod() / 2;
}

inline void app_main(
        smarties::Communicator*const comm, // communicator with smarties
        MPI_Comm mpicom,                  // mpi_comm that mpi-based apps can use
        int argc, char**argv             // args read from app's runtime settings file
) {
    for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}
    #ifdef STEFANS_SENSORS_STATE
    const int nActions = 2, nStates = 16;
    comm->setStateActionDims(nStates, nActions);
    std::vector<bool> b_observable =  //{ false, false, false, false, false, false, false, false, false, false, //blind fish
            { true, true, true, true, true, true, true, true, true, true, //vision
                    //  false, false, false, false, false, false }; // no shear
              true, true, true, true, true, true }; //shear
    comm->setStateObservable(b_observable);
    #else
    const int nActions = 2, nStates = 10;
    comm->setStateActionDims(nStates, nActions);
    #endif
    const unsigned maxLearnStepPerSim = 200; // random number... TODO

    // Tell smarties that action space should be bounded.
    // First action modifies curvature, only makes sense between -1 and 1
    // Second action affects Tp = (1+act[1])*Tperiod_0 (eg. halved if act[1]=-.5).
    // If too small Re=L^2*Tp/nu would increase too much, we allow it to
    //  double at most, therefore we set the bounds between -0.5 and 0.5.
    std::vector<double> upper_action_bound{1.,.25}, lower_action_bound{-1.,-.25};
    comm->setActionScales(upper_action_bound, lower_action_bound, true);

    Simulation sim(argc, argv);
    sim.init();

    StefanFish*const agent = dynamic_cast<StefanFish*>( sim.getShapes()[0] );
    if(agent==nullptr) { printf("Agent was not a StefanFish!\n"); abort(); }

    if(comm->isTraining() == false) {
        sim.sim.verbose = true; sim.sim.muteAll = false;
        sim.sim.dumpTime = agent->Tperiod / 20;
    }
    unsigned sim_id = 0, tot_steps = 0;

    // Terminate loop if reached max number of time steps. Never terminate if 0
    while( true ) // train loop
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

        double t = 0, tNextAct = 0;
        unsigned step = 0;
        bool agentOver = false;

        comm->sendInitState( agent->getCStartState() ); //send initial state

        while (true) //simulation loop
        {
            setAction(agent, comm->recvAction(), tNextAct);
            tNextAct = getTimeToNextAct(agent, tNextAct);
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
            std::vector<double> state = agent->getCStartState();
            double reward = getReward(agent, t);

            if (agentOver || checkNaN(state, reward) ) {
                printf("Agent failed\n"); fflush(0);
                comm->sendTermState(state, reward);
                break;
            }
            else
            if (step >= maxLearnStepPerSim) {
                printf("Sim ended\n"); fflush(0);
                comm->sendLastState(state, reward);
                break;
            }
            else comm->sendState(state, reward);
        } // simulation is done

        if(comm->isTraining() == false) chdir("../");
        sim_id++;

        if (comm->terminateTraining()) return; // exit program
    }
}

int main(int argc, char**argv)
{
    smarties::Engine e(argc, argv);
    if( e.parse() ) return 1;
    e.run( app_main );
    return 0;
}
