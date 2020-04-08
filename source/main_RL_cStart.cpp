/* main_RL_cStart.cpp
 * Created by Ioannis Mandralis (ioannima@ethz.ch)
 * Main script for a single CStartFish learning a fast start
*/

#include <unistd.h>
#include <sys/stat.h>

#include "smarties.h"
#include "Simulation.h"
#include "Obstacles/CStartFish.h"

using namespace cubism;

inline void resetIC(CStartFish* const a, smarties::Communicator*const c)
{
    double com[2] = {0.5, 0.5};
    a->setCenterOfMass(com);
    a->setOrientation(-98.0 * M_PI / 180.0);
}

inline void setAction(CStartFish* const agent,
                      const std::vector<double> act, const double t)
{
    agent->act(t, act);
}

inline bool isTerminal(const CStartFish*const a, const Real& time) {
    // Terminate when the fish exits a radius of 1.5 characteristic lengths
    printf("Time of current episode is: %f\n", time);
    return (a->getRadialDisplacement() >= 1.50 * a->length) || time > 2.0 ;
}

inline double getReward(const CStartFish* const a, const double& t_elapsed) {
    // Reward is negative the time between each action
    // time dependent
    // distance from initial condition.
    // terminal reward: total time it took
    // minimize energy ? would use the motion with maximum energy...
    // (efficiency)
    // only control the curvature
    double radialDisplacement = a->getRadialDisplacement();
    double reward = isTerminal(a, t_elapsed)? radialDisplacement - t_elapsed : radialDisplacement + a->EffPDefBnd;
    printf("Stage reward is: %f \n", reward);
    return reward;

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

inline void app_main(
        smarties::Communicator*const comm, // communicator with smarties
        MPI_Comm mpicom,                  // mpi_comm that mpi-based apps can use
        int argc, char**argv             // args read from app's runtime settings file
) {
    printf("In app_main\n");
    // Define the maximum learn steps per simulation (episode)
    const unsigned maxLearnStepPerSim = 200; // not sure how to set this

    for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}
    const int nActions = 8, nStates = 14;
    comm->setStateActionDims(nStates, nActions);

    Simulation sim(argc, argv);
    sim.init();
    printf("Simulation initialized\n");

    CStartFish*const agent = dynamic_cast<CStartFish*>( sim.getShapes()[0] );
    if(agent==nullptr) { printf("Agent was not a CStartFish!\n"); abort(); }
    printf("Agent initialized\n");

    std::vector<double> lower_action_bound{-4, -1, -1, -6, -3, -1.5, 0, 0}, upper_action_bound{0, 0, 0, 0, 0, 0, +1, +1};
    comm->setActionScales(upper_action_bound, lower_action_bound, true);

    if(comm->isTraining() == false) {
        sim.sim.verbose = true; sim.sim.muteAll = false;
        sim.sim.dumpTime = agent->Tperiod / 20;
    }

    unsigned int sim_id = 0, tot_steps = 0;

    printf("Entering train loop\n");
    // Terminate loop if reached max number of time steps. Never terminate if 0
    while( true ) // train loop
    {
        printf("In train loop\n");
        if(comm->isTraining() == false)
        {
            char dirname[1024]; dirname[1023] = '\0';
            sprintf(dirname, "run_%08u/", sim_id);
            printf("Starting a new sim in directory %s\n", dirname);
            mkdir(dirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            chdir(dirname);
        }

        printf("Resetting sim\n");
        sim.reset();
        printf("Resetting IC\n");
        resetIC(agent, comm); // randomize initial conditions

        double t = 0, tNextAct = 0;
        unsigned int step = 0;
        bool agentOver = false;

        printf("Sending initial state\n");
        comm->sendInitState( agent->state() ); //send initial state

        printf("Entering simulation loop\n");
        while (true) //simulation loop
        {
            printf("Setting action\n");
            setAction(agent, comm->recvAction(), tNextAct);
            tNextAct = agent->getTimeNextAct();
            printf("tNextAct: %f\n", tNextAct);

            while (t < tNextAct)
            {
                const double dt = sim.calcMaxTimestep();
                printf("dt: %f\n", dt);
                t += dt;
                printf("t: %f\n", t);

                if ( sim.advance( dt ) ) { // if true sim has ended
                    printf("Set -tend 0. This file decides the length of train sim.\n");
                    assert(false); fflush(0); abort();
                }
                printf("step smarties is %d\n", step);
                printf("sim.time is %f\n", t);
                if ( isTerminal(agent, t)) {
                    agentOver = true;
                    break;
                }
            }
            step++;
            tot_steps++;
            std::vector<double> state = agent->state();
            double reward = getReward(agent, t);

            if (agentOver || checkNaN(state, reward)) {
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
