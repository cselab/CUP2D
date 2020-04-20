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

#if 0
inline void resetIC(CStartFish* const a, smarties::Communicator*const c)
{
    double com[2] = {0.5, 0.5};
    a->setCenterOfMass(com);
    a->setOrientation(-98.0 * M_PI / 180.0);
}
#endif
#if 1
inline void resetIC(CStartFish* const a, smarties::Communicator*const c, double target[2])
{
    double com[2] = {0.5, 0.5};
    double initialAngle = -98.0;
    a->setCenterOfMass(com);
    a->setOrientation(initialAngle * M_PI / 180.0);
    // Place target at 1.5 fish lengths away (1.5 * 0.25 = 0.375). Diametrically opposite from initial orientation.
    double fishLengthRadius = 1.5;
    double length = a->length;
    double supplementaryAngle = 180.0 + initialAngle;
    target[0] = fishLengthRadius * length * std::cos(supplementaryAngle);
    target[1] = fishLengthRadius * length * std::sin(supplementaryAngle);
}
#endif

inline void setAction(CStartFish* const agent,
                      const std::vector<double> act, const double t)
{
    agent->act(t, act);
}

#if 0
inline bool isTerminal(const CStartFish*const a, const Real& time) {
    // Terminate when the fish exits a radius of 1.5 characteristic lengths
    printf("Time of current episode is: %f\n", time);
    return (a->getRadialDisplacement() >= 1.50 * a->length) || time > 2.0 ;
}
#endif

#if 1
inline bool isTerminal(const CStartFish*const a, const Real& time) {
    // Terminate after time equal to Gazzola's c-start
    printf("Time of current episode is: %f\n", time);
    return time > 1.5882352941 ;
}
#endif

#if 0
inline double getReward(const CStartFish* const a, const double& t_elapsed, const double& energyExpended) {

    // Baseline energy consumed by a C-start:
    const double baselineEnergy = 0.011; // in joules
    // Relative difference in energy of current policy with respect to normal C-start:
    double relativeChangeEnergy = std::abs(energyExpended - baselineEnergy) / baselineEnergy;
    // Dimensionless radial displacement:
    double dimensionlessRadialDisplacement = a->getRadialDisplacement() / a->length;
    // Dimensionless episode time:
    double dimensionlessTElapsed = t_elapsed / a->Tperiod;

    // Stage reward
    double stageReward = dimensionlessRadialDisplacement;
    // Terminal reward
    double terminalReward = dimensionlessRadialDisplacement - relativeChangeEnergy - dimensionlessTElapsed;
    // Overall reward
    double reward = isTerminal(a, t_elapsed)? terminalReward : stageReward;

    printf("Stage reward is: %f \n", reward);
    return reward;
}
#endif
#if 0
inline double getReward(const CStartFish* const a, const double& t_elapsed) {

    // Dimensionless radial displacement:
    double dimensionlessRadialDisplacement = a->getRadialDisplacement() / a->length;
    // Dimensionless episode time:
    double dimensionlessTElapsed = t_elapsed / a->Tperiod;

    // Stage reward
    double stageReward = dimensionlessRadialDisplacement;
    // Terminal reward
    double terminalReward = dimensionlessRadialDisplacement - dimensionlessTElapsed;
    // Overall reward
    double reward = isTerminal(a, t_elapsed)? terminalReward : stageReward;

    printf("Stage reward is: %f \n", reward);
    return reward;
}
#endif

#if 0
inline double getReward(const CStartFish* const a) {
    // Dimensionless radial displacement:
    double dimensionlessRadialDisplacement = a->getRadialDisplacement() / a->length;
    // Reward is dimensionless radial displacement:
    double reward = dimensionlessRadialDisplacement;
    printf("Stage reward is: %f \n", reward);
    return reward;
}
#endif

#if 1
inline double getReward(const CStartFish* const a, double previousRelativePosition[2]) {
    // Reward inspired from Zermelo's problem (without the penalty per time step)

    // Current position and previous position relative to target in absolute coordinates.
    double relativeX = a->state()[0] * a->length;
    double relativeY = a->state()[1] * a->length;
//    double prevRelativeX = previousRelativePosition[0] * a->length;
//    double prevRelativeY = previousRelativePosition[1] * a->length;

    // Current distance and previous distance from target in absolute units.
    double distance_ = (std::sqrt(std::pow(relativeX, 2) + std::pow(relativeY, 2))) / a->length;
//    double prevDistance_ = (std::sqrt(std::pow(prevRelativeX, 2) + std::pow(prevRelativeY, 2))) / a->length;

//    double reward = 1/distance_ - 1/prevDistance_;

    // Simpler reward structure
//    double radialDisplacement_ = a->getRadialDisplacement() / a->length;
    double reward = -distance_;

    printf("Stage reward is: %f \n", reward);
    return reward;
}
#endif

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
    // Define the maximum learn steps per simulation (episode)
    const unsigned maxLearnStepPerSim = 200; // not sure how to set this

    for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}
    const int nActions = 8, nStates = 14;
    comm->setStateActionDims(nStates, nActions);

    Simulation sim(argc, argv);
    sim.init();

    CStartFish*const agent = dynamic_cast<CStartFish*>( sim.getShapes()[0] );
    if(agent==nullptr) { printf("Agent was not a CStartFish!\n"); abort(); }

//    std::vector<double> lower_action_bound{-4, -1, -1, -6, -3, -1.5, 0, 0}, upper_action_bound{0, 0, 0, 0, 0, 0, +1, +1};
    std::vector<double> lower_action_bound{-4, -1, -1, -6, -3, -1.5, 0, 0}, upper_action_bound{+4, +1, +1, 0, 0, 0, +1, +1};
    comm->setActionScales(upper_action_bound, lower_action_bound, true);

    if(comm->isTraining() == false) {
        sim.sim.verbose = true; sim.sim.muteAll = false;
        sim.sim.dumpTime = agent->Tperiod / 20;
    }

    unsigned int sim_id = 0, tot_steps = 0;

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

        double target[2] = {0.0, 0.0}; // initialize target
        resetIC(agent, comm, target); // randomize initial conditions
        printf("Target[0] is %f, Target[1] is %f\n", target[0], target[1]);
        agent->setTarget(target); // set the target

        double outTarget[2] = {0.0, 0.0};
        agent->getTarget(outTarget);
        printf("Target[0] is %f, Target[1] is %f\n", outTarget[0], outTarget[1]);


        double t = 0, tNextAct = 0;
        unsigned int step = 0;
        bool agentOver = false;
        double previousRelativePosition[2] = {0.0, 0.0};

//        // Energy consumed by fish in one episode.
//        double energyExpended = 0.0;

        comm->sendInitState( agent->state() ); //send initial state
        while (true) //simulation loop
        {
            setAction(agent, comm->recvAction(), tNextAct);
            tNextAct = agent->getTimeNextAct();

            // Store the previous relative position before advancing
            previousRelativePosition[0] = agent->state()[0];
            previousRelativePosition[1] = agent->state()[1];
            printf("Previous relative position is (%f, %f)", previousRelativePosition[0], previousRelativePosition[1]);

            while (t < tNextAct)
            {
                const double dt = sim.calcMaxTimestep();
                t += dt;

//                printf("Get the power output\n");
//                energyExpended += -agent->defPowerBnd * dt; // We want work done by fish on fluid.

                if ( sim.advance( dt ) ) { // if true sim has ended
                    printf("Set -tend 0. This file decides the length of train sim.\n");
                    assert(false); fflush(0); abort();
                }

                if ( isTerminal(agent, t)) {
                    agentOver = true;
                    break;
                }
            }
            step++;
            tot_steps++;
            std::vector<double> state = agent->state();
            printf("state is (%f, %f, ...)", state[0], state[1]);
//            printf("Energy expended is: %f\n", energyExpended);
//            double reward = getReward(agent, t, energyExpended);
//            double reward = getReward(agent, t);
            double reward = getReward(agent, previousRelativePosition);

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
