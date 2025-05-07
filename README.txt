Pairwise Information Sharing Algorithm for Wildfire Surveillance
----------------------------------------------------------------

This program runs a wildfire simulation in a 25x25 grid with swarming UAVs tracking the fire front. Green cells are healthy trees, red cells are burning/on fire trees, and the black cell is the center of the fire spread. UAVs are represented as blue circles.

To run the file, use the command 'python3 .\sim.py' in terminal.

There are several functions that run different experiments to analyze the effect of the various key parameters: wind and swarm size. To choose which functions to run, simply comment/uncomment the desired function calls.

You can run the simulation with visualization by including 'runsim(render=True)' in the main loop.

Plots can be found in the 'Results' folder.