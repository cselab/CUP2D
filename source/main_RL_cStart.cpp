/* main_RL_cStart.cpp
 * Created by Ioannis Mandralis (ioannima@ethz.ch)
 * Main script for a single CStartFish learning a fast start. Task based reasoning for a smarties application.
*/

#include <unistd.h>
#include <sys/stat.h>
#include "smarties.h"
#include "Simulation.h"
#include "Obstacles/CStartFish.h"


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
    inline void setAction(CStartFish* const agent,
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

// Target following
class GoToTarget : public Task
{
public:
    // Task constants
    std::vector<double> lower_action_bound{-4, -1, -1, -6, -3, -1.5, 0, 0};
    std::vector<double> upper_action_bound{+4, +1, +1, 0, 0, 0, +1, +1};
    const int nActions = 8;
    const int nStates = 15;
    unsigned maxActionsPerSim = 9000000;
public:
    inline void resetIC(CStartFish* const a, smarties::Communicator*const c)
    {
        double initialAngle = -98.0;
        double length = a->length;
        double com[2] = {0.5, 0.5};
        double target[2] = {0.0, 0.0};


        // Place target at 1.5 fish lengths away (1.5 * 0.25 = 0.375).
        double targetRadius_ = 1.5;
        double targetRadius = targetRadius_ * length;
//    double targetAngle = 180.0 + initialAngle; // Diametrically opposite from initial orientation.
        double targetAngle = initialAngle; // Directly ahead of fish
        target[0] = targetRadius * std::cos(targetAngle);
        target[1] = targetRadius * std::sin(targetAngle);

        // Set agent quantities
        a->setCenterOfMass(com);
        a->setOrientation(initialAngle * M_PI / 180.0);
        a->setTarget(target);
    }

    inline bool isTerminal(const CStartFish*const a) {
        // Terminate after time equal to Gazzola's C-start
        printf("Time of current episode is: %f\n", timeElapsed);
        return timeElapsed > 1.5882352941 ;
    }

    inline std::vector<double> getState(const CStartFish* const a)
    {
        return a->stateTarget();
    }

    inline double getReward(const CStartFish* const a)
    {
        return -getState(a)[0];
    }

    inline double getTerminalReward(const CStartFish* const a)
    {
        return getReward(a);
    }

};

// Escaping
class Escape : public Task
{
public:
    // Task constants
    std::vector<double> lower_action_bound{-2*M_PI, -2*M_PI, -2*M_PI, -2*M_PI, -2*M_PI, -2*M_PI, 0, 0, 0};
    std::vector<double> upper_action_bound{0, 0, 0, 0, 0, 0, +1, +1, +1};
    int nActions = 9;
    int nStates = 25;
    unsigned maxActionsPerSim = 9000000;
public:

    inline void resetIC(CStartFish* const a, smarties::Communicator*const c)
    {
        const Real A = 10*M_PI/180; // start between -10 and 10 degrees
        std::uniform_real_distribution<Real> dis(-A, A);
        const auto SA = c->isTraining() ? dis(c->getPRNG()) : -10.0 * M_PI / 180.0;
        a->setOrientation(SA);
        double com[2] = {0.7, 0.5};
        a->setCenterOfMass(com);
        double vo[2] = {0.9, 0.5};
        a->setVirtualOrigin(vo);
    }

    inline bool isTerminal(const CStartFish*const a)
    {
        return timeElapsed > 1.5882352941;
    }

    inline std::vector<double> getState(const CStartFish* const a)
    {
        return a->stateEscape();
    }

};
class DistanceEscape : public Escape
{
public:
    inline double getReward(const CStartFish* const a)
    {
        return 0.0;
    }
    inline double getTerminalReward(const CStartFish* const a)
    {
        return a->getRadialDisplacement() / a->length;
    }
};
class DistanceEnergyEscape : public Escape
{
public:
    const double baselineEnergy = 0.00730; // Baseline energy consumed by a C-start in joules for 0.93 lengths in 16x16 res
public:
    inline double getReward(const CStartFish* const a)
    {
        return 0.0;
    }
    inline double getTerminalReward(const CStartFish* const a)
    {
        const double polarAngle = a->getPolarAngle();
        const bool outsidePolarSweep = std::abs(polarAngle) < 160* M_PI/180.0;
        const double orientation = a->getOrientation();
        const bool orientationOutOfRange = std::abs(orientation) >= 90* M_PI/180.0;

        const double penaltyPolar = outsidePolarSweep ? -10 : 0;
        const double penaltyOrientation = orientationOutOfRange ? -10 : 0;

//        printf("[terminalReward] reward is %f\n", a->getRadialDisplacement() / a->length + penaltyPolar + penaltyOrientation);
        return a->getRadialDisplacement() / a->length + penaltyPolar + penaltyOrientation;
    }
    inline bool isTerminal(const CStartFish*const a)
    {
        const double polarAngle = a->getPolarAngle();
        const bool outsidePolarSweep = std::abs(polarAngle) < 160* M_PI/180.0;
        const double orientation = a->getOrientation();
        const bool orientationOutOfRange = std::abs(orientation) >= 90* M_PI/180.0;

//        printf("[isTerminal] polarAngle %f energyExpended %f outsidePolarSweep %d orientationOutOfRange %d\n", polarAngle, energyExpended, outsidePolarSweep, orientationOutOfRange);
//        printf("[isTerminal] radialDisplacementState %f polarAngleState %f energyExpendedState %f orientationState %f\n", a->stateEscape()[0], a->stateEscape()[1], a->stateEscape()[2], a->stateEscape()[3]);
        return (timeElapsed > 1.5882352941 || energyExpended > baselineEnergy || outsidePolarSweep || orientationOutOfRange);
    }

};
class DistanceVariableEnergyEscape : public Escape
{
public:
    double baselineEnergy = 0.0; // Baseline energy consumed by a C-start in joules for 0.93 lengths in 16x16 res
public:
    inline double getReward(const CStartFish* const a)
    {
        return 0.0;
    }
    inline double getTerminalReward(const CStartFish* const a)
    {
        const double polarAngle = a->getPolarAngle();
        const bool outsidePolarSweep = std::abs(polarAngle) < 160* M_PI/180.0;
        const double orientation = a->getOrientation();
        const bool orientationOutOfRange = std::abs(orientation) >= 90* M_PI/180.0;
        const double penaltyPolar = outsidePolarSweep ? -10 : 0;
        const double penaltyOrientation = orientationOutOfRange ? -10 : 0;
        return a->getRadialDisplacement() / a->length + penaltyPolar + penaltyOrientation;
    }
    inline bool isTerminal(const CStartFish*const a)
    {
        const double polarAngle = a->getPolarAngle();
        const bool outsidePolarSweep = std::abs(polarAngle) < 160* M_PI/180.0;
        const double orientation = a->getOrientation();
        const bool orientationOutOfRange = std::abs(orientation) >= 90* M_PI/180.0;
        return (timeElapsed > 1.5882352941 || energyExpended > baselineEnergy || outsidePolarSweep || orientationOutOfRange);
    }
    inline void resetIC(CStartFish* const a, smarties::Communicator*const c)
    {
        const Real A = 10*M_PI/180; // start between -10 and 10 degrees
        std::uniform_real_distribution<Real> dis(-A, A);
        const auto SA = c->isTraining() ? dis(c->getPRNG()) : -10.0 * M_PI / 180.0;
        a->setOrientation(SA);
        double com[2] = {0.7, 0.5};
        a->setCenterOfMass(com);
        double vo[2] = {0.9, 0.5};
        a->setVirtualOrigin(vo);
        std::uniform_real_distribution<Real> disEnergy(0.00243, 0.0219);
//        baselineEnergy = c->isTraining() ? disEnergy(c->getPRNG()) : 0.0219;
        baselineEnergy = disEnergy(c->getPRNG());
        a->setEnergyBudget(baselineEnergy);
//        printf("[resetIC] agent energy budget is %f\n", a->getEnergyBudget());
    }
    inline std::vector<double> getState(const CStartFish* const a)
    {
//        printf("[getState] agent energy diff is %f\n", a->stateEscapeVariableEnergy()[2]);
//        printf("[getState] agent energy expended is %f\n", a->getEnergyExpended());
        return a->stateEscapeVariableEnergy();

    }

};
class SequentialDistanceEnergyEscape : public Escape
{
public:
    const double baselineEnergy = 0.00730; // Baseline energy consumed by a C-start in joules for 0.93 lengths in 16x16 res
public:
    inline double getReward(const CStartFish* const a)
    {
        const double polarAngle = a->getPolarAngle();
        const bool outsidePolarSweep = std::abs(polarAngle) < 160* M_PI/180.0;
        const double orientation = a->getOrientation();
        const bool orientationOutOfRange = std::abs(orientation) >= 90* M_PI/180.0;
        const double penaltyPolar = outsidePolarSweep ? -10 : 0;
        const double penaltyOrientation = orientationOutOfRange ? -10 : 0;
        return getState(a)[0] + penaltyPolar + penaltyOrientation;
    }
    inline double getTerminalReward(const CStartFish* const a)
    {
        return getReward(a);
    }
    inline bool isTerminal(const CStartFish*const a)
    {
        const double polarAngle = a->getPolarAngle();
        const bool outsidePolarSweep = std::abs(polarAngle) < 160* M_PI/180.0;
        const double orientation = a->getOrientation();
        const bool orientationOutOfRange = std::abs(orientation) >= 90* M_PI/180.0;
        return (timeElapsed > 1.5882352941 || outsidePolarSweep || orientationOutOfRange);

    }
    inline std::vector<double> getState(const CStartFish* const a)
    {
        return a->stateSequentialEscape();
    }
};


class SequentialDistanceEscape : public Escape
{
public:
    inline double getReward(const CStartFish* const a)
    {
        return a->getRadialDisplacement() / a->length;
    }
    inline double getTerminalReward(const CStartFish* const a)
    {
        return getReward(a);
    }
};
class SequentialDistanceTimeEscape : public Escape
{
public:
    inline double getReward(const CStartFish* const a)
    {
        // Dimensionless radial displacement:
        return a->getRadialDisplacement() / a->length;
    }
    inline double getTerminalReward(const CStartFish* const a)
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
    inline double getReward(const CStartFish* const a)
    {
        // Dimensionless radial displacement:
        return a->getRadialDisplacement() / a->length;
    }
    inline double getTerminalReward(const CStartFish* const a)
    {
        // Add the relative difference in energy of current policy with respect to normal C-start:
        return a->getRadialDisplacement() / a->length - std::abs(energyExpended - baselineEnergy) / baselineEnergy - timeElapsed / a->Tperiod;
    }
};

// Escape tradeoff
class EscapeTradeoff : public Task
{
public:
    // Task constants
    std::vector<double> lower_action_bound{-2*M_PI, -2*M_PI, -2*M_PI, -2*M_PI, -2*M_PI, -2*M_PI, 0, 0, 0};
    std::vector<double> upper_action_bound{0, 0, 0, 0, 0, 0, +1, +1, +1};
    int nActions = 9;
    int nStates = 26;
    unsigned maxActionsPerSim = 9000000;
    const double baselineEnergy = 0.00730; // Baseline energy consumed by a C-start in joules for 0.93 lengths in 16x16 res
public:

    inline void resetIC(CStartFish* const a, smarties::Communicator*const c)
    {
        const Real A = 10*M_PI/180; // start between -10 and 10 degrees
        std::uniform_real_distribution<Real> dis(-A, A);
        const auto SA = c->isTraining() ? dis(c->getPRNG()) : -10.0 * M_PI / 180.0;
        a->setOrientation(SA);
        double com[2] = {0.7, 0.5};
        a->setCenterOfMass(com);
        double vo[2] = {0.9, 0.5};
        a->setVirtualOrigin(vo);
        // Set the energy budget
        a->setEnergyBudget(baselineEnergy);
    }

    inline bool isTerminal(const CStartFish*const a)
    {
        return timeElapsed > 1.5882352941;
    }

    inline std::vector<double> getState(const CStartFish* const a)
    {
        return a->stateEscapeTradeoff();
    }

};
class DistanceTradeoffEnergyEscape : public EscapeTradeoff
{
public:
    const double relativeSpeedFactor = 2.70;
public:
    inline double getReward(const CStartFish* const a)
    {
        return 0.0;
    }
    inline double getTerminalReward(const CStartFish* const a)
    {
        const double polarAngle = a->getPolarAngle();
        const bool outsidePolarSweep = std::abs(polarAngle) < 160* M_PI/180.0;
        const double orientation = a->getOrientation();
        const bool orientationOutOfRange = std::abs(orientation) >= 90* M_PI/180.0;

        const double penaltyPolar = outsidePolarSweep ? -10 : 0;
        const double penaltyOrientation = orientationOutOfRange ? -10 : 0;

        // Get the distance that the fish had travelled at Tprop

//        printf("[terminalReward] reward is %f\n", a->getRadialDisplacement() / a->length + relativeSpeedFactor * getState(a)[1] + penaltyPolar + penaltyOrientation);
//        printf("[terminalReward] rewardTradeoff is %f\n", relativeSpeedFactor * getState(a)[1]);
        return a->getRadialDisplacement() / a->length + relativeSpeedFactor * getState(a)[1] + penaltyPolar + penaltyOrientation;
    }
    inline bool isTerminal(const CStartFish*const a)
    {
        const double polarAngle = a->getPolarAngle();
        const bool outsidePolarSweep = std::abs(polarAngle) < 160* M_PI/180.0;
        const double orientation = a->getOrientation();
        const bool orientationOutOfRange = std::abs(orientation) >= 90* M_PI/180.0;

//        printf("[isTerminal] polarAngle %f energyExpended %f outsidePolarSweep %d orientationOutOfRange %d\n", polarAngle, energyExpended, outsidePolarSweep, orientationOutOfRange);
//        printf("[isTerminal] radialDisplacementState %f dTprop % f polarAngleState %f energyExpendedState %f orientationState %f\n", getState(a)[0], getState(a)[1], getState(a)[2], getState(a)[3], getState(a)[4]);
        return (timeElapsed > 1.5882352941 || energyExpended > baselineEnergy || outsidePolarSweep || orientationOutOfRange);
    }

};


// CStart
class CStart : public Task
{
public:
    // Task constants
    unsigned maxActionsPerSim = 2;
    std::vector<double> lower_action_bound{-4, -1, -1, -6, -3, -1.5, 0};
    std::vector<double> upper_action_bound{0, 0, 0, 0, 0, 0, +1};
    int nActions = 7;
    int nStates = 14;
public:
    inline void resetIC(CStartFish* const a, smarties::Communicator*const c)
    {
        const Real A = 10*M_PI/180; // start between -10 and 10 degrees
        std::uniform_real_distribution<Real> dis(-A, A);
        const auto SA = c->isTraining() ? dis(c->getPRNG()) : -90.0 * M_PI / 180.0;
        a->setOrientation(SA);
        double com[2] = {0.5, 0.5};
        a->setCenterOfMass(com);
    }
    inline bool isTerminal(const CStartFish*const a)
    {
        return timeElapsed > 1.5882352941 ;
    }
    inline std::vector<double> getState(const CStartFish* const a)
    {
        return a->stateEscape();
    }
    inline void setAction(CStartFish* const agent, const std::vector<double> act, const double t)
    {
        agent->actCStart(t, act);
    }
    inline double getReward(const CStartFish* const a)
    {
        return 0.0;
    }
    inline double getTerminalReward(const CStartFish* const a)
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
    SequentialDistanceEnergyEscape task = SequentialDistanceEnergyEscape();

    // Inform smarties communicator of the task
    for(int i=0; i<argc; i++) {printf("arg: %s\n",argv[i]); fflush(0);}
    comm->setStateActionDims(task.nStates, task.nActions);
    comm->setActionScales(task.upper_action_bound, task.lower_action_bound, true);
    const unsigned maxLearnStepPerSim = task.maxLearnStepPerSim;

    // Initialize the simulation
    Simulation sim(argc, argv);
    sim.init();

    CStartFish* const agent = dynamic_cast<CStartFish*>( sim.getShapes()[0] );
    if(agent==nullptr) { printf("Agent was not a CStartFish!\n"); abort(); }
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
        double energyExpended = 0.0; // Energy consumed by fish in one episode

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
                energyExpended += -agent->defPowerBnd * dt; // We want work done by fish on fluid.
                agent->setEnergyExpended(energyExpended);
//                if (t <= agent->Tperiod) { agent->setDistanceTprop(agent->getRadialDisplacement()); }
//                printf("radial disp %f\n", agent->getRadialDisplacement());

                // Set the task-energy-expenditure
                task.setEnergyExpended(energyExpended);

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
