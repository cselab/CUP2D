/* main_RL_cStart.cpp
 * Created by Ioannis Mandralis (ioannima@ethz.ch)
 * Main script for a single ZebraFish learning a fast start. Task based reasoning for a smarties application.
*/

#include <unistd.h>
#include <sys/stat.h>
#include "smarties.h"
#include "Simulation.h"
#include "Obstacles/ZebraFish.h"


using namespace cubism;

// A task is something that shares the same agent, state, action-set, terminal condition, and initial condition.
class Task
{
public:
    const unsigned maxLearnStepPerSim = 9000000;
    // Simulation quantities needed for reward functions
    double timeElapsed = 0.0; // time elapsed until now in the episode
    double energyExpended = 0.0; // energy expended until now in the episode
public:
    inline void setAction(ZebraFish* const agent,
                          const std::vector<double> act, const double t)
    {
        agent->act(t, act);
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

    inline void setTimeElapsed(const double& t) {
        timeElapsed = t;
    }

    inline void setEnergyExpended(const double& e) {
        energyExpended = e;
    }

};

// Escaping
class Escape : public Task
{
public:
    // Task constants
    std::vector<double> lower_action_bound{-1, 0, 0, 0};
    std::vector<double> upper_action_bound{-1, 1, 1, 1};
    int nActions = 4;
    int nStates = 10;
    unsigned maxActionsPerSim = 9000000;
public:

    inline void resetIC(ZebraFish* const a, smarties::Communicator*const c)
    {
        const Real A = 10*M_PI/180; // start between -10 and 10 degrees
        std::uniform_real_distribution<Real> dis(-A, A);
        const auto SA = c->isTraining() ? dis(c->getPRNG()) : -98.0 * M_PI / 180.0;
        a->setOrientation(SA);
        double com[2] = {0.5, 0.5};
        a->setCenterOfMass(com);
    }

    inline bool isTerminal(const ZebraFish*const a)
    {
        // Terminate when the fish exits a radius of 1.5 characteristic lengths or time > 2.0
        return (a->getRadialDisplacement() >= 1.50 * a->length) || timeElapsed > 2.0 ;
    }

    inline std::vector<double> getState(const ZebraFish* const a)
    {
        return a->state();
    }

};
class DistanceEscape : public Escape
{
public:
    inline bool isTerminal(const ZebraFish*const a)
    {
        return timeElapsed > 1.5882352941 ;
    }
    inline double getReward(const ZebraFish* const a)
    {
        return 0.0;
    }
    inline double getTerminalReward(const ZebraFish* const a)
    {
        return a->getRadialDisplacement() / a->length;
    }
};
class SequentialDistanceEscape : public Escape
{
public:
    inline double getReward(const ZebraFish* const a)
    {
        return a->getRadialDisplacement() / a->length;
    }
    inline double getTerminalReward(const ZebraFish* const a)
    {
        return getReward(a);
    }
};
class SequentialDistanceTimeEscape : public Escape
{
public:
    inline double getReward(const ZebraFish* const a)
    {
        // Dimensionless radial displacement:
        return a->getRadialDisplacement() / a->length;
    }
    inline double getTerminalReward(const ZebraFish* const a)
    {
        // Dimensionless radial displacement and time elapsed:
        return a->getRadialDisplacement() / a->length - timeElapsed / a->Tperiod;
    }
};
class SequentialDistanceTimeEnergyEscape : public Escape
{
public:
    const double baselineEnergy = 0.011; // Baseline energy consumed by a C-start in joules
public:
    inline double getReward(const ZebraFish* const a)
    {
        // Dimensionless radial displacement:
        return a->getRadialDisplacement() / a->length;
    }
    inline double getTerminalReward(const ZebraFish* const a)
    {
        // Add the relative difference in energy of current policy with respect to normal C-start:
        return a->getRadialDisplacement() / a->length - std::abs(energyExpended - baselineEnergy) / baselineEnergy - timeElapsed / a->Tperiod;
    }
};

// CStart
class CStart : public Task
{
public:
    // Task constants
    unsigned maxActionsPerSim = 3;
    std::vector<double> lower_action_bound{-1, 0, 0, 0};
    std::vector<double> upper_action_bound{-1, 1, 1, 1};
    int nActions = 4;
    int nStates = 10;
public:
    inline void resetIC(ZebraFish* const a, smarties::Communicator*const c)
    {
        const Real A = 10*M_PI/180; // start between -10 and 10 degrees
        std::uniform_real_distribution<Real> dis(-A, A);
        const auto SA = c->isTraining() ? dis(c->getPRNG()) : -90.0 * M_PI / 180.0;
        a->setOrientation(SA);
        double com[2] = {0.5, 0.5};
        a->setCenterOfMass(com);
    }
    inline bool isTerminal(const ZebraFish*const a)
    {
        return timeElapsed > 1.5882352941 ;
    }
    inline std::vector<double> getState(const ZebraFish* const a)
    {
        return a->state();
    }
    inline void setAction(ZebraFish* const agent, const std::vector<double> act, const double t)
    {
        agent->act(t, act);
    }
    inline double getReward(const ZebraFish* const a)
    {
        return 0.0;
    }
    inline double getTerminalReward(const ZebraFish* const a)
    {
        return a->getRadialDisplacement() / a->length;
    }
};

inline void app_main(
        smarties::Communicator*const comm, // communicator with smarties
        MPI_Comm mpicom,                   // mpi_comm that mpi-based apps can use
        int argc, char**argv               // args read from app's runtime settings file
) {
    // Get the task definition
    CStart task = CStart();

    // Inform smarties communicator of the task
    for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}
    comm->setStateActionDims(task.nStates, task.nActions);
    comm->setActionScales(task.upper_action_bound, task.lower_action_bound, true);
    const unsigned maxLearnStepPerSim = task.maxLearnStepPerSim;

    // Initialize the simulation
    Simulation sim(argc, argv);
    sim.init();

    ZebraFish* const agent = dynamic_cast<ZebraFish*>( sim.getShapes()[0] );
    if(agent==nullptr) { printf("Agent was not a ZebraFish!\n"); abort(); }
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
        task.resetIC(agent, comm); // randomize initial conditions

        double t = 0, tNextAct = 0;
        unsigned step = 0, numActions = 0;
        bool agentOver = false;
        // double energyExpended = 0.0; // Energy consumed by fish in one episode

        comm->sendInitState( task.getState(agent) ); //send initial state

        while (true) //simulation loop
        {
            if (numActions < task.maxActionsPerSim) {
                task.setAction(agent, comm->recvAction(), tNextAct);
                tNextAct = agent->getTimeNextAct();
                numActions += 1;
            } else {
                tNextAct = 10.0;
                // wait for episode to end. 10 seconds is alot. This only works for tasks which are time bounded.
            }

            while (t < tNextAct)
            {
                const double dt = sim.calcMaxTimestep();
                t += dt; task.setTimeElapsed(t); // set the task time.

                // Forward integrate the energy expenditure
                // energyExpended += -agent->defPowerBnd * dt; // We want work done by fish on fluid.

                // Set the task-energy-expenditure
                // task.setEnergyExpended(energyExpended);

                if ( sim.advance( dt ) ) { // if true sim has ended
                    printf("Set -tend 0. This file decides the length of train sim.\n");
                    assert(false); fflush(0); abort();
                }

                if ( task.isTerminal(agent)) {
                    agentOver = true;
                    break;
                }
            }
            step++;
            tot_steps++;
            std::vector<double> state = task.getState(agent);
            double reward = agentOver? task.getTerminalReward(agent) : task.getReward(agent);

            if (agentOver || task.checkNaN(state, reward)) {
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