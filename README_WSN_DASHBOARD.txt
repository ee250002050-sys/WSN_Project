WSN Lifetime Optimization Dashboard
==================================

Files
-----
topology_editor_component/index.html  Drag-and-drop graph editor
wsn_dashboard.py                       Main Streamlit dashboard
wsn_lifetime_lp.py                     LP model, radio-energy model and baseline
requirements.txt                       Python dependencies

Setup (macOS)
-------------
1. Keep all files and the topology_editor_component folder together.
2. Open Terminal in this project folder.
3. (Recommended) Create and activate a virtual environment:

   python3 -m venv .venv
   source .venv/bin/activate

4. Install dependencies:

   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

5. Start the dashboard:

   streamlit run wsn_dashboard.py

6. Open http://localhost:8501 in your browser.

Network architecture
--------------------
The network contains n_sensors ordinary sensor nodes, one additional
battery-powered node labelled SOURCE, and one SINK. Every ordinary sensor
continues to generate its own measurement traffic, and the designated SOURCE
also generates traffic. Both ordinary sensors and SOURCE can forward traffic
and have battery constraints. The SINK is the sole destination and does not
transmit. Thus the displayed SOURCE is a real LP node, not just a visual tag.

Main features
-------------
- Five tabs: Overview, Network Simulator, Energy Simulation,
  Optimization & Analysis, Experiments & Reports.
- Drag nodes, add links by clicking two nodes, and remove links by clicking
  an edge in the custom topology editor.
- Designated SOURCE (orange hexagon), ordinary sensor nodes (blue circles),
  and SINK (purple diamond).
- Per-node generation-rate controls and communication-range filtering.
- LP versus shortest-path comparison using the same radio-energy model.
- Average-flow battery depletion playback with Play/Pause/Reset and speed
  options up to 100x.
- LP results are cached during playback so each refresh does not resolve the
  same optimization problem again.
- Per-node TX/RX energy, bottleneck and battery shadow-price analysis.
- Complementary-slackness and LP primal-dual diagnostics.
- Multi-seed, network-density and battery-scale experiments.
- CSV and JSON exports.

Model limitations
-----------------
The animation uses average traffic rates and average power derived from the
routing solution. It is not a packet-level simulator: it does not model MAC
scheduling, packet collisions, retransmissions, channel fading, or hardware.
The shortest-path baseline sends each generating node's own traffic along its
minimum-transmission-energy path; the LP may split flows across paths.

Playback speed
--------------
The animation refreshes at a target 100 ms interval. At 100x, it advances by
up to 10 simulated seconds per refresh. Browser, network and Streamlit rendering
overhead may make the achieved speed lower than the selected target.

Validation
----------
The Python scripts pass syntax compilation. The LP and shortest-path solvers
were run on sample networks with 5, 10, 20 and 50 ordinary sensors, each with
an additional SOURCE node and a SINK. The Streamlit UI still needs to be
launched locally and tested with the installed dependency versions before a
live project demonstration.
