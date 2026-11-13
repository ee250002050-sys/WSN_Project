WSN Lifetime Optimization Dashboard
==================================

Files
-----
topology_editor_component/index.html  Drag-and-drop graph editor
wsn_dashboard.py                       Main Streamlit dashboard
wsn_lifetime_lp.py                     LP model, radio-energy model and baseline
wsn_packet_sim.py                      Discrete-time packet-level simulator
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
and have battery constraints. The SINK is the sole destination. Thus the
SOURCE is a real LP node, not just a visual tag.

Main features
-------------
- Five tabs: Overview, Network Simulator, Energy Simulation,
  Optimization & Analysis, and Experiments & Reports.
- Drag nodes, add links by clicking two nodes, and remove links by clicking
  an edge in the custom topology editor.
- Designated SOURCE (orange hexagon), ordinary sensor nodes (blue circles),
  and SINK (purple diamond).
- Per-node generation-rate controls and communication-range filtering.
- LP versus shortest-path comparison using the same radio-energy model.
- Average-flow battery depletion playback with Play/Pause/Reset and speed
  options up to 100x.
- Per-node TX/RX energy, bottleneck and battery shadow-price analysis.
- Complementary-slackness and LP primal-dual diagnostics.
- Multi-seed, network-density and battery-scale experiments.

The existing average-flow animation remains an LP-derived average-power
model; it is not replaced by the packet simulator. The shortest-path baseline
in the LP comparison sends each generating node's own traffic along its
minimum-transmission-energy path; the LP may split flows across paths.

Playback speed
--------------
The average-flow animation refreshes at a target 100 ms interval. At 100x, it
advances by up to 10 simulated seconds per refresh. Browser, network and
Streamlit rendering overhead may make the achieved speed lower than the
selected target.

Validation
----------
The Python scripts pass syntax compilation. The LP solver and packet
simulator were run on sample networks; a selected-link removal changed packet
outcomes on tested route-critical links. A live Streamlit UI launch could not
be performed in the build environment because Streamlit is not installed
there. Install requirements.txt and launch locally before a live demonstration.
